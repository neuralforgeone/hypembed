//! Model module (internal).
//!
//! Implements a BERT-like encoder architecture:
//! configuration parsing, SafeTensors loading, embeddings, attention, FFN, pooling.

pub(crate) mod attention;
pub(crate) mod config;
pub(crate) mod embedding;
pub(crate) mod encoder;
pub(crate) mod ff;
pub(crate) mod pool;
pub(crate) mod safetensors;
pub(crate) mod weights;
