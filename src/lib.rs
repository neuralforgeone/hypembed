#![allow(clippy::too_many_arguments)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::needless_range_loop)]
//! # HypEmbed
//!
//! A pure-Rust, local-first embedding inference library.
//!
//! HypEmbed loads transformer model weights, tokenizes text, runs a forward pass,
//! and returns normalized embedding vectors — all in safe Rust with zero external
//! ML runtime dependencies.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};
//!
//! let model = Embedder::load("./model").unwrap();
//! let options = EmbeddingOptions::default()
//!     .with_normalize(true)
//!     .with_pooling(PoolingStrategy::Mean);
//! let embeddings = model.embed(&["hello world", "rust embeddings"], &options).unwrap();
//! ```

pub mod error;
pub mod tensor;
pub mod tokenizer;
pub mod model;
pub mod pipeline;

// Public re-exports for ergonomic API
pub use pipeline::embedder::{Embedder, EmbeddingOptions};
pub use model::pool::PoolingStrategy;
pub use error::HypEmbedError;
