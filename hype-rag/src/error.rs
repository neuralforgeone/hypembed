use thiserror::Error;

pub type Result<T> = std::result::Result<T, HypeRagError>;

#[derive(Debug, Error)]
pub enum HypeRagError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Embedding error: {0}")]
    Embed(#[from] hypembed::HypEmbedError),
    #[error("{0}")]
    Other(String),
}