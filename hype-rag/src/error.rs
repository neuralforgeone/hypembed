//! Error types for the hype-rag library.

use thiserror::Error;

/// Result alias for hype-rag operations.
pub type Result<T> = std::result::Result<T, HypeRagError>;

/// Typed failures for indexing, search, and storage.
#[derive(Debug, Error)]
pub enum HypeRagError {
    /// Filesystem IO failure.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// SQLite persistence failure.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// JSON serialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Embedding inference failure from hypembed.
    #[error("Embedding error: {0}")]
    Embed(#[from] hypembed::HypEmbedError),

    /// Caller supplied invalid input (empty index, bad paths, etc.).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Missing or incomplete configuration (model dir, data dir).
    #[error("Configuration error: {0}")]
    Config(String),

    /// Catch-all for operational failures with context.
    #[error("{0}")]
    Other(String),
}

impl HypeRagError {
    /// Build a [`HypeRagError::InvalidInput`] error.
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Build a [`HypeRagError::Config`] error.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }
}
