//! Error types for the HypEmbed library.
//!
//! All public functions return [`Result`] with [`HypEmbedError`] on failure.
//! Variants carry enough context to diagnose load, tokenization, and inference issues
//! without inspecting stack traces.

use thiserror::Error;

/// The main error type for HypEmbed.
///
/// Variants are grouped by failure domain. Prefer [`HypEmbedError::InvalidInput`],
/// [`HypEmbedError::Config`], and [`HypEmbedError::Tokenization`] for user-facing paths;
/// lower-level [`HypEmbedError::Tensor`] and [`HypEmbedError::Model`] variants surface
/// numerical and weight-loading failures.
#[derive(Error, Debug)]
pub enum HypEmbedError {
    /// Caller supplied invalid or unsupported input.
    #[error("Invalid input: {context}")]
    InvalidInput {
        /// Human-readable description of what was wrong with the input.
        context: String,
    },

    /// Model or runtime configuration is missing, inconsistent, or unsupported.
    #[error("Configuration error ({field}): {message}")]
    Config {
        /// Config field or subsystem (e.g. `hidden_size`, `model_dir`).
        field: String,
        /// What was wrong with that field.
        message: String,
    },

    /// Text tokenization failed at a specific pipeline stage.
    #[error("Tokenization failed at {stage}: {detail}")]
    Tokenization {
        /// Pipeline stage (e.g. `vocab`, `encode`, `wordpiece`).
        stage: String,
        /// Stage-specific failure detail.
        detail: String,
    },

    /// Tensor operation error (shape mismatch, invalid index, etc.).
    #[error("Tensor error: {0}")]
    Tensor(String),

    /// Tokenizer error (vocab loading, encoding, etc.).
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Model error (weight loading, config parsing, forward pass).
    #[error("Model error: {0}")]
    Model(String),

    /// IO error (file not found, read failure).
    #[error("IO error: {source}")]
    Io {
        /// Underlying IO failure.
        #[from]
        source: std::io::Error,
    },

    /// JSON parsing error.
    #[error("JSON error: {source}")]
    Json {
        /// Underlying JSON parse failure.
        #[from]
        source: serde_json::Error,
    },
}

impl HypEmbedError {
    /// Build an [`HypEmbedError::InvalidInput`] error.
    pub fn invalid_input(context: impl Into<String>) -> Self {
        Self::InvalidInput {
            context: context.into(),
        }
    }

    /// Build an [`HypEmbedError::Config`] error.
    pub fn config(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Config {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Build an [`HypEmbedError::Tokenization`] error.
    pub fn tokenization(stage: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Tokenization {
            stage: stage.into(),
            detail: detail.into(),
        }
    }
}

/// Convenience result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, HypEmbedError>;
