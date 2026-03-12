/// Model configuration.
///
/// Parsed from a `config.json` file (HuggingFace format).
/// Defines the architecture hyperparameters of a BERT-like model.

use serde::Deserialize;
use std::path::Path;
use crate::error::{HypEmbedError, Result};

/// Configuration for a BERT-like encoder model.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Size of the vocabulary (number of token embeddings).
    pub vocab_size: usize,

    /// Dimensionality of the hidden states.
    pub hidden_size: usize,

    /// Number of encoder layers.
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    pub num_attention_heads: usize,

    /// Dimensionality of the feed-forward intermediate layer.
    pub intermediate_size: usize,

    /// Maximum number of position embeddings.
    pub max_position_embeddings: usize,

    /// Layer normalization epsilon.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,

    /// Activation function name (e.g., "gelu").
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Number of token type (segment) embeddings.
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,

    /// Model type (e.g., "bert", "distilbert", "xlm-roberta").
    /// Auto-detected from config.json.
    #[serde(default)]
    pub model_type: Option<String>,
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

fn default_hidden_act() -> String {
    "gelu".to_string()
}

fn default_type_vocab_size() -> usize {
    2
}

impl ModelConfig {
    /// Load configuration from a `config.json` file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: ModelConfig = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(HypEmbedError::Model("hidden_size must be > 0".into()));
        }
        if self.num_attention_heads == 0 {
            return Err(HypEmbedError::Model("num_attention_heads must be > 0".into()));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(HypEmbedError::Model(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        Ok(())
    }

    /// Head dimension = hidden_size / num_attention_heads.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Layer norm eps as f32.
    pub fn ln_eps(&self) -> f32 {
        self.layer_norm_eps as f32
    }

    /// Check if this is a DistilBERT model.
    pub fn is_distilbert(&self) -> bool {
        self.model_type.as_deref() == Some("distilbert")
    }

    /// Check if this is a standard BERT-type model (BERT, MiniLM, etc.).
    pub fn is_bert(&self) -> bool {
        !self.is_distilbert()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 1536,
            "max_position_embeddings": 512
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        config.validate().unwrap();
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.head_dim(), 32);
        assert_eq!(config.ln_eps(), 1e-12);
    }

    #[test]
    fn test_invalid_config() {
        let json = r#"{
            "vocab_size": 100,
            "hidden_size": 10,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "intermediate_size": 40,
            "max_position_embeddings": 128
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert!(config.validate().is_err()); // 10 % 3 != 0
    }
}
