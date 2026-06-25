//! Local RAG toolkit powered by hypembed.

pub mod chunk;
pub mod error;
pub mod index;
pub mod search;
pub mod store;
pub mod text_boundary;

pub use chunk::{chunk_text, Chunk};
pub use error::{HypeRagError, Result};
pub use index::index_directory;
pub use search::{cosine_similarity, search_chunks};
pub use store::ChunkStore;
