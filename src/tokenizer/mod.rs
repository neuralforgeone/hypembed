//! Tokenizer module (internal).
//!
//! Implements a BERT-compatible tokenizer pipeline:
//! 1. Pre-tokenization (lowercasing, whitespace/punctuation splitting)
//! 2. WordPiece subword tokenization
//! 3. Encoding (adding special tokens, truncation, padding)

pub(crate) mod encode;
pub(crate) mod pre_tokenize;
pub(crate) mod vocab;
pub(crate) mod wordpiece;

pub(crate) use encode::Tokenizer;
