#![allow(clippy::too_many_arguments)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]
// Internal tensor/model utilities include mmap loaders and math helpers not yet on every path.
#![allow(dead_code)]
//! # HypEmbed
//!
//! A pure-Rust, local-first embedding inference library.
//!
//! HypEmbed loads transformer model weights, tokenizes text, runs a forward pass,
//! and returns normalized embedding vectors — all in safe Rust with zero external
//! ML runtime dependencies.
//!
//! ## Public API
//!
//! The stable surface is intentionally small:
//!
//! - [`Embedder`] — load a model and generate embeddings
//! - [`EmbeddingOptions`] — pooling, normalization, and sequence length
//! - [`PoolingStrategy`] — mean or CLS pooling
//! - [`HypEmbedError`] — typed failures with actionable context
//!
//! Low-level tensor, tokenizer, and model modules are `pub(crate)` and not part of the stable contract.
//!
//! ## Quick Start (directory)
//!
//! ```rust,no_run
//! use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};
//!
//! let model = Embedder::load("./model").unwrap();
//! let options = EmbeddingOptions::default()
//!     .with_normalize(true)
//!     .with_pooling(PoolingStrategy::Mean);
//! let embeddings = model.embed(&["hello world", "rust embeddings"], &options).unwrap();
//! assert_eq!(embeddings.len(), 2);
//! ```
//!
//! ## Quick Start (in-memory bytes)
//!
//! ```
//! use hypembed::{Embedder, EmbeddingOptions, PoolingStrategy};
//!
//! # mod fixtures {
//! #   include!("../tests/common/mod.rs");
//! # }
//! # use fixtures::*;
//! # fn main() -> Result<(), hypembed::HypEmbedError> {
//! let embedder = Embedder::from_bytes(
//!     TINY_CONFIG_JSON,
//!     &tiny_vocab_txt(),
//!     &tiny_safetensors_bytes(),
//!     true,
//! )?;
//! let options = EmbeddingOptions::default()
//!     .with_normalize(true)
//!     .with_pooling(PoolingStrategy::Mean)
//!     .with_max_length(16);
//! let embeddings = embedder.embed(&["hello world"], &options)?;
//! assert_eq!(embeddings.len(), 1);
//! assert_eq!(embeddings[0].len(), embedder.hidden_size());
//! # Ok(())
//! # }
//! ```

/// Error types and the [`HypEmbedError`] result alias.
pub mod error;

pub(crate) mod model;
pub(crate) mod pipeline;
pub(crate) mod tensor;
pub(crate) mod tokenizer;

pub use error::HypEmbedError;
pub use model::pool::PoolingStrategy;
pub use pipeline::embedder::{Embedder, EmbeddingOptions};
