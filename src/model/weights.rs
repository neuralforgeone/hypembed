/// Weight loading and management.
///
/// Loads model weights from a SafeTensors file, maps them to the correct
/// layer structure, and validates shapes against the model configuration.
///
/// ## v0.2: Multi-model support
///
/// Supports BERT, MiniLM (BERT-compatible), and DistilBERT weight naming.
/// DistilBERT uses different tensor names and lacks token_type_embeddings.

use std::path::Path;
use crate::error::{HypEmbedError, Result};
use crate::tensor::Tensor;
use crate::model::config::ModelConfig;
use crate::model::safetensors::SafeTensorsFile;

/// Weights for a single attention layer.
#[derive(Debug)]
pub struct AttentionWeights {
    pub query_weight: Tensor,
    pub query_bias: Tensor,
    pub key_weight: Tensor,
    pub key_bias: Tensor,
    pub value_weight: Tensor,
    pub value_bias: Tensor,
    pub output_weight: Tensor,
    pub output_bias: Tensor,
    pub output_ln_weight: Tensor,
    pub output_ln_bias: Tensor,
}

/// Weights for a single feed-forward layer.
#[derive(Debug)]
pub struct FeedForwardWeights {
    pub intermediate_weight: Tensor,
    pub intermediate_bias: Tensor,
    pub output_weight: Tensor,
    pub output_bias: Tensor,
    pub output_ln_weight: Tensor,
    pub output_ln_bias: Tensor,
}

/// Weights for a single encoder layer.
#[derive(Debug)]
pub struct EncoderLayerWeights {
    pub attention: AttentionWeights,
    pub ff: FeedForwardWeights,
}

/// All model weights.
#[derive(Debug)]
pub struct ModelWeights {
    pub word_embeddings: Tensor,
    pub position_embeddings: Tensor,
    /// `None` for models that don't use token type embeddings (e.g., DistilBERT).
    pub token_type_embeddings: Option<Tensor>,
    pub embedding_ln_weight: Tensor,
    pub embedding_ln_bias: Tensor,
    pub layers: Vec<EncoderLayerWeights>,
}

impl ModelWeights {
    /// Load weights from a SafeTensors file.
    ///
    /// Maps HuggingFace BERT naming convention to our weight structure.
    pub fn load<P: AsRef<Path>>(path: P, config: &ModelConfig) -> Result<Self> {
        let st = SafeTensorsFile::load(path)?;
        Self::from_safetensors(&st, config)
    }

    /// Load weights from a SafeTensors file using memory-mapped I/O.
    pub fn load_mmap<P: AsRef<Path>>(path: P, config: &ModelConfig) -> Result<Self> {
        let st = SafeTensorsFile::load_mmap(path)?;
        Self::from_safetensors(&st, config)
    }

    /// Load weights from a parsed SafeTensors file.
    fn from_safetensors(st: &SafeTensorsFile, config: &ModelConfig) -> Result<Self> {
        if config.is_distilbert() {
            Self::load_distilbert(st, config)
        } else {
            Self::load_bert(st, config)
        }
    }

    /// Load BERT/MiniLM weights (standard HuggingFace naming).
    fn load_bert(st: &SafeTensorsFile, config: &ModelConfig) -> Result<Self> {
        // Embedding weights — try with and without "bert." prefix
        let word_embeddings = st.get_tensor("embeddings.word_embeddings.weight")
            .or_else(|_| st.get_tensor("bert.embeddings.word_embeddings.weight"))?;
        let position_embeddings = st.get_tensor("embeddings.position_embeddings.weight")
            .or_else(|_| st.get_tensor("bert.embeddings.position_embeddings.weight"))?;
        let token_type_embeddings = st.get_tensor("embeddings.token_type_embeddings.weight")
            .or_else(|_| st.get_tensor("bert.embeddings.token_type_embeddings.weight"))?;
        let embedding_ln_weight = st.get_tensor("embeddings.LayerNorm.weight")
            .or_else(|_| st.get_tensor("bert.embeddings.LayerNorm.weight"))?;
        let embedding_ln_bias = st.get_tensor("embeddings.LayerNorm.bias")
            .or_else(|_| st.get_tensor("bert.embeddings.LayerNorm.bias"))?;

        // Validate embedding shapes
        Self::validate_shape(&word_embeddings, &[config.vocab_size, config.hidden_size], "word_embeddings")?;
        Self::validate_shape(&position_embeddings, &[config.max_position_embeddings, config.hidden_size], "position_embeddings")?;
        Self::validate_shape(&token_type_embeddings, &[config.type_vocab_size, config.hidden_size], "token_type_embeddings")?;

        // Load each encoder layer
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Self::load_bert_layer(st, config, i)?;
            layers.push(layer);
        }

        Ok(ModelWeights {
            word_embeddings,
            position_embeddings,
            token_type_embeddings: Some(token_type_embeddings),
            embedding_ln_weight,
            embedding_ln_bias,
            layers,
        })
    }

    /// Load DistilBERT weights.
    ///
    /// DistilBERT differences from BERT:
    /// - No token_type_embeddings
    /// - Different tensor naming: `distilbert.embeddings.*`, `distilbert.transformer.layer.*`
    /// - Attention: `q_lin`, `k_lin`, `v_lin`, `out_lin` instead of `self.query`, etc.
    /// - LayerNorm names: `sa_layer_norm`, `output_layer_norm`
    fn load_distilbert(st: &SafeTensorsFile, config: &ModelConfig) -> Result<Self> {
        let word_embeddings = st.get_tensor("distilbert.embeddings.word_embeddings.weight")
            .or_else(|_| st.get_tensor("embeddings.word_embeddings.weight"))?;
        let position_embeddings = st.get_tensor("distilbert.embeddings.position_embeddings.weight")
            .or_else(|_| st.get_tensor("embeddings.position_embeddings.weight"))?;
        let embedding_ln_weight = st.get_tensor("distilbert.embeddings.LayerNorm.weight")
            .or_else(|_| st.get_tensor("embeddings.LayerNorm.weight"))?;
        let embedding_ln_bias = st.get_tensor("distilbert.embeddings.LayerNorm.bias")
            .or_else(|_| st.get_tensor("embeddings.LayerNorm.bias"))?;

        Self::validate_shape(&word_embeddings, &[config.vocab_size, config.hidden_size], "word_embeddings")?;
        Self::validate_shape(&position_embeddings, &[config.max_position_embeddings, config.hidden_size], "position_embeddings")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Self::load_distilbert_layer(st, config, i)?;
            layers.push(layer);
        }

        Ok(ModelWeights {
            word_embeddings,
            position_embeddings,
            token_type_embeddings: None,
            embedding_ln_weight,
            embedding_ln_bias,
            layers,
        })
    }

    fn load_bert_layer(st: &SafeTensorsFile, config: &ModelConfig, layer_idx: usize) -> Result<EncoderLayerWeights> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;

        // Try both naming conventions: with and without "bert." prefix
        let get = |short_name: &str| -> Result<Tensor> {
            let full = format!("encoder.layer.{}.{}", layer_idx, short_name);
            let bert_full = format!("bert.encoder.layer.{}.{}", layer_idx, short_name);
            st.get_tensor(&full).or_else(|_| st.get_tensor(&bert_full))
        };

        let attention = AttentionWeights {
            query_weight: get("attention.self.query.weight")?,
            query_bias: get("attention.self.query.bias")?,
            key_weight: get("attention.self.key.weight")?,
            key_bias: get("attention.self.key.bias")?,
            value_weight: get("attention.self.value.weight")?,
            value_bias: get("attention.self.value.bias")?,
            output_weight: get("attention.output.dense.weight")?,
            output_bias: get("attention.output.dense.bias")?,
            output_ln_weight: get("attention.output.LayerNorm.weight")?,
            output_ln_bias: get("attention.output.LayerNorm.bias")?,
        };

        Self::validate_shape(&attention.query_weight, &[h, h], "query_weight")?;
        Self::validate_shape(&attention.query_bias, &[h], "query_bias")?;

        let ff = FeedForwardWeights {
            intermediate_weight: get("intermediate.dense.weight")?,
            intermediate_bias: get("intermediate.dense.bias")?,
            output_weight: get("output.dense.weight")?,
            output_bias: get("output.dense.bias")?,
            output_ln_weight: get("output.LayerNorm.weight")?,
            output_ln_bias: get("output.LayerNorm.bias")?,
        };

        Self::validate_shape(&ff.intermediate_weight, &[inter, h], "intermediate_weight")?;
        Self::validate_shape(&ff.output_weight, &[h, inter], "output_weight")?;

        Ok(EncoderLayerWeights { attention, ff })
    }

    fn load_distilbert_layer(st: &SafeTensorsFile, config: &ModelConfig, layer_idx: usize) -> Result<EncoderLayerWeights> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;

        let get = |short_name: &str| -> Result<Tensor> {
            let full = format!("distilbert.transformer.layer.{}.{}", layer_idx, short_name);
            let short = format!("transformer.layer.{}.{}", layer_idx, short_name);
            st.get_tensor(&full).or_else(|_| st.get_tensor(&short))
        };

        // DistilBERT uses q_lin, k_lin, v_lin, out_lin
        let attention = AttentionWeights {
            query_weight: get("attention.q_lin.weight")?,
            query_bias: get("attention.q_lin.bias")?,
            key_weight: get("attention.k_lin.weight")?,
            key_bias: get("attention.k_lin.bias")?,
            value_weight: get("attention.v_lin.weight")?,
            value_bias: get("attention.v_lin.bias")?,
            output_weight: get("attention.out_lin.weight")?,
            output_bias: get("attention.out_lin.bias")?,
            // DistilBERT: sa_layer_norm is the post-attention LayerNorm
            output_ln_weight: get("sa_layer_norm.weight")?,
            output_ln_bias: get("sa_layer_norm.bias")?,
        };

        Self::validate_shape(&attention.query_weight, &[h, h], "query_weight")?;

        let ff = FeedForwardWeights {
            intermediate_weight: get("ffn.lin1.weight")?,
            intermediate_bias: get("ffn.lin1.bias")?,
            output_weight: get("ffn.lin2.weight")?,
            output_bias: get("ffn.lin2.bias")?,
            // DistilBERT: output_layer_norm is the post-FFN LayerNorm
            output_ln_weight: get("output_layer_norm.weight")?,
            output_ln_bias: get("output_layer_norm.bias")?,
        };

        Self::validate_shape(&ff.intermediate_weight, &[inter, h], "intermediate_weight")?;
        Self::validate_shape(&ff.output_weight, &[h, inter], "output_weight")?;

        Ok(EncoderLayerWeights { attention, ff })
    }

    fn validate_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
        if tensor.shape().dims() != expected {
            return Err(HypEmbedError::Model(format!(
                "Shape mismatch for '{}': expected {:?}, got {}",
                name, expected, tensor.shape()
            )));
        }
        Ok(())
    }
}
