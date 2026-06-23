/// Vocabulary for BERT-style models.
///
/// Loads a `vocab.txt` file where each line is a token, and the line number
/// (0-indexed) is the token's ID. This is the standard format used by
/// BERT, DistilBERT, MiniLM, etc.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use crate::error::{HypEmbedError, Result};

/// Special token constants.
pub const CLS_TOKEN: &str = "[CLS]";
pub const SEP_TOKEN: &str = "[SEP]";
pub const PAD_TOKEN: &str = "[PAD]";
pub const UNK_TOKEN: &str = "[UNK]";
pub const MASK_TOKEN: &str = "[MASK]";

/// Vocabulary mapping between tokens and IDs.
#[derive(Debug, Clone)]
pub struct Vocab {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl Vocab {
    /// Parse vocabulary from `vocab.txt` content.
    ///
    /// Each line is a token. Line number (0-indexed) is the token ID.
    pub fn from_str(content: &str) -> Result<Self> {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        for (idx, line) in content.lines().enumerate() {
            let token = line.to_string();
            token_to_id.insert(token.clone(), idx as u32);
            id_to_token.push(token);
        }

        if id_to_token.is_empty() {
            return Err(HypEmbedError::Tokenizer("Vocab file is empty".into()));
        }

        for special in &[CLS_TOKEN, SEP_TOKEN, PAD_TOKEN, UNK_TOKEN] {
            if !token_to_id.contains_key(*special) {
                return Err(HypEmbedError::Tokenizer(format!(
                    "Missing required special token '{}' in vocab",
                    special
                )));
            }
        }

        Ok(Self {
            token_to_id,
            id_to_token,
        })
    }

    /// Load vocabulary from a `vocab.txt` file.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            HypEmbedError::Tokenizer(format!(
                "Failed to read vocab file '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Self::from_str(&content)
    }

    /// Look up a token's ID. Returns UNK ID if not found.
    pub fn token_id(&self, token: &str) -> u32 {
        self.token_to_id
            .get(token)
            .copied()
            .unwrap_or_else(|| self.unk_id())
    }

    /// Check if a token exists in the vocabulary.
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Get the ID of the `[UNK]` token.
    pub fn unk_id(&self) -> u32 {
        self.token_to_id[UNK_TOKEN]
    }

    /// Get the ID of the `[CLS]` token.
    pub fn cls_id(&self) -> u32 {
        self.token_to_id[CLS_TOKEN]
    }

    /// Get the ID of the `[SEP]` token.
    pub fn sep_id(&self) -> u32 {
        self.token_to_id[SEP_TOKEN]
    }

    /// Get the ID of the `[PAD]` token.
    pub fn pad_id(&self) -> u32 {
        self.token_to_id[PAD_TOKEN]
    }

    /// Get number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Get token string by ID.
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }
}
