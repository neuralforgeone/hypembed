use crate::error::Result;
use crate::model::config::ModelConfig;
use crate::model::embedding;
use crate::model::encoder;
use crate::model::pool::{self, PoolingStrategy};
use crate::model::weights::ModelWeights;
use crate::tensor::normalize;
use crate::tokenizer::Tokenizer;
/// The main embedding pipeline.
///
/// `Embedder` is the primary user-facing type. It combines tokenization,
/// model inference, pooling, and normalization into a single API.
///
/// # Usage
///
/// ```rust,no_run
/// use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};
///
/// let model = Embedder::load("./model").unwrap();
/// let options = EmbeddingOptions::default()
///     .with_normalize(true)
///     .with_pooling(PoolingStrategy::Mean);
/// let embeddings = model.embed(&["hello world"], &options).unwrap();
/// ```

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// Options controlling pooling, normalization, and sequence length for [`Embedder::embed`].
///
/// Use the builder-style helpers ([`with_max_length`](Self::with_max_length),
/// [`with_pooling`](Self::with_pooling), [`with_normalize`](Self::with_normalize)) to override defaults.
///
/// # Examples
///
/// ```
/// use hypembed::{EmbeddingOptions, PoolingStrategy};
///
/// let options = EmbeddingOptions::default()
///     .with_max_length(64)
///     .with_pooling(PoolingStrategy::Cls)
///     .with_normalize(false);
/// assert_eq!(options.max_length, 64);
/// assert!(!options.normalize);
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingOptions {
    /// Maximum sequence length (including `[CLS]` and `[SEP]`).
    /// Default: 128
    pub max_length: usize,
    /// Pooling strategy. Default: Mean
    pub pooling: PoolingStrategy,
    /// Whether to L2-normalize the output. Default: true
    pub normalize: bool,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            max_length: 128,
            pooling: PoolingStrategy::Mean,
            normalize: true,
        }
    }
}

impl EmbeddingOptions {
    /// Set the maximum sequence length.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the pooling strategy.
    pub fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Set whether to L2-normalize the output.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// The main embedder. Combines tokenizer, model, and post-processing.
///
/// Load from a directory ([`load`](Self::load)) or in-memory bytes ([`from_bytes`](Self::from_bytes)),
/// then call [`embed`](Self::embed) to produce one vector per input string.
///
/// `Embedder` is [`Send`] + [`Sync`] and may be shared across threads when wrapped in `Arc`.
pub struct Embedder {
    tokenizer: Tokenizer,
    config: ModelConfig,
    weights: ModelWeights,
}

impl Embedder {
    /// Load a model entirely from in-memory bytes (no filesystem access).
    ///
    /// # Arguments
    /// - `config_json`: UTF-8 JSON model configuration
    /// - `vocab_txt`: UTF-8 vocab.txt content
    /// - `weights_bytes`: Raw SafeTensors file bytes
    /// - `do_lower_case`: Whether to lowercase input text during tokenization
    pub fn from_bytes(
        config_json: &str,
        vocab_txt: &str,
        weights_bytes: &[u8],
        do_lower_case: bool,
    ) -> Result<Self> {
        let config = ModelConfig::from_json_str(config_json)?;
        let vocab = vocab_txt.parse::<crate::tokenizer::vocab::Vocab>()?;
        let tokenizer = Tokenizer::from_vocab(vocab, do_lower_case);
        let weights = ModelWeights::from_bytes(weights_bytes, &config)?;

        Ok(Self {
            tokenizer,
            config,
            weights,
        })
    }

    /// Load a model from a directory.
    ///
    /// The directory must contain:
    /// - `config.json` — model configuration
    /// - `vocab.txt` — tokenizer vocabulary
    /// - `model.safetensors` — model weights
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let dir = model_dir.as_ref();

        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(crate::error::HypEmbedError::config(
                "model_dir",
                format!("config.json not found in '{}'", dir.display()),
            ));
        }
        let config_json = std::fs::read_to_string(&config_path)?;

        let vocab_path = dir.join("vocab.txt");
        if !vocab_path.exists() {
            return Err(crate::error::HypEmbedError::config(
                "model_dir",
                format!("vocab.txt not found in '{}'", dir.display()),
            ));
        }
        let vocab_txt = std::fs::read_to_string(&vocab_path)?;

        let weights_path = dir.join("model.safetensors");
        if !weights_path.exists() {
            return Err(crate::error::HypEmbedError::config(
                "model_dir",
                format!("model.safetensors not found in '{}'", dir.display()),
            ));
        }
        let weights_bytes = std::fs::read(&weights_path)?;

        Self::from_bytes(&config_json, &vocab_txt, &weights_bytes, true)
    }

    /// Generate embeddings for one or more texts.
    ///
    /// # Arguments
    /// - `texts`: Slice of text strings to embed
    /// - `options`: Embedding options (pooling, normalization, max_length)
    ///
    /// # Returns
    /// A vector of embedding vectors, one per input text.
    /// Each embedding is a `Vec<f32>` of length `hidden_size`.
    pub fn embed(&self, texts: &[&str], options: &EmbeddingOptions) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Step 1: Tokenize
        let encodings = self.tokenizer.encode_batch(texts, options.max_length)?;

        let input_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.input_ids.clone()).collect();
        let attention_mask: Vec<Vec<u32>> =
            encodings.iter().map(|e| e.attention_mask.clone()).collect();
        let token_type_ids: Vec<Vec<u32>> =
            encodings.iter().map(|e| e.token_type_ids.clone()).collect();

        // Step 2: Embeddings
        let hidden = embedding::compute_embeddings(
            &input_ids,
            &token_type_ids,
            &self.weights.word_embeddings,
            &self.weights.position_embeddings,
            self.weights.token_type_embeddings.as_ref(),
            &self.weights.embedding_ln_weight,
            &self.weights.embedding_ln_bias,
            self.config.ln_eps(),
        )?;

        // Step 3: Encoder forward
        let hidden =
            encoder::encoder_forward(hidden, &attention_mask, &self.weights.layers, &self.config)?;

        // Step 4: Pooling
        let pooled = pool::pool(&hidden, &attention_mask, options.pooling)?;

        // Step 5: Optional L2 normalization
        let output = if options.normalize {
            normalize::l2_normalize(&pooled, 1e-12)?
        } else {
            pooled
        };

        // Step 6: Convert to Vec<Vec<f32>>
        let batch_size = texts.len();
        let hidden_size = self.config.hidden_size;
        let data = output.data();

        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let start = b * hidden_size;
            let end = start + hidden_size;
            result.push(data[start..end].to_vec());
        }

        Ok(result)
    }

    /// Get the model's hidden size (embedding dimension).
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pool::PoolingStrategy;

    mod tiny_fixtures {
        include!("../../tests/common/mod.rs");
    }

    fn tiny_embedder() -> Embedder {
        use tiny_fixtures::*;
        Embedder::from_bytes(
            TINY_CONFIG_JSON,
            &tiny_vocab_txt(),
            &tiny_safetensors_bytes(),
            true,
        )
        .expect("tiny embedder")
    }

    #[test]
    fn embedder_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Embedder>();
        assert_send_sync::<EmbeddingOptions>();
        assert_send_sync::<PoolingStrategy>();
    }

    #[test]
    fn embed_empty_text_batch_returns_empty() {
        let embedder = tiny_embedder();
        let options = EmbeddingOptions::default();
        let out = embedder.embed(&[], &options).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn embed_unicode_and_special_chars() {
        let embedder = tiny_embedder();
        let options = EmbeddingOptions::default().with_max_length(16);
        let out = embedder
            .embed(&["café 🦀 日本語", "[CLS]"], &options)
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), embedder.hidden_size());
    }

    #[test]
    fn embed_truncation_at_short_max_length() {
        let embedder = tiny_embedder();
        let options = EmbeddingOptions::default().with_max_length(4);
        let long = "hello world rust embedding machine learning semantic search";
        let out = embedder.embed(&[long], &options).unwrap();
        assert_eq!(out[0].len(), embedder.hidden_size());
    }

    #[test]
    fn embed_cls_pooling_without_normalization() {
        let embedder = tiny_embedder();
        let options = EmbeddingOptions::default()
            .with_pooling(PoolingStrategy::Cls)
            .with_normalize(false)
            .with_max_length(16);
        let out = embedder.embed(&["hello"], &options).unwrap();
        let norm: f32 = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0);
        assert!(
            (norm - 1.0).abs() > 1e-3,
            "CLS without normalize should not be unit length"
        );
    }
}
