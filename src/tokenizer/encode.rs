/// Encoding: the final tokenizer output.
///
/// Combines pre-tokenization, WordPiece, and special token handling
/// into a complete encoding pipeline.

use std::path::Path;
use crate::error::{HypEmbedError, Result};
use crate::tokenizer::vocab::Vocab;
use crate::tokenizer::wordpiece;
use crate::tokenizer::pre_tokenize;
use rayon::prelude::*;

/// The output of tokenizing a single text.
#[derive(Debug, Clone)]
pub struct Encoding {
    /// Token IDs (integers indexing into the vocabulary).
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<u32>,
    /// Token type IDs (0 for first segment, 1 for second).
    /// For single-sentence tasks, all zeros.
    pub token_type_ids: Vec<u32>,
}

/// BERT-compatible tokenizer.
///
/// Combines vocabulary, pre-tokenization, and WordPiece into a single pipeline.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vocab,
    do_lower_case: bool,
}

impl Tokenizer {
    /// Create a new tokenizer from a vocabulary file.
    ///
    /// # Arguments
    /// - `vocab_path`: Path to `vocab.txt`
    /// - `do_lower_case`: Whether to lowercase input text
    pub fn new<P: AsRef<Path>>(vocab_path: P, do_lower_case: bool) -> Result<Self> {
        let vocab = Vocab::load(vocab_path)?;
        Ok(Self {
            vocab,
            do_lower_case,
        })
    }

    /// Create a tokenizer from a pre-loaded vocabulary.
    pub fn from_vocab(vocab: Vocab, do_lower_case: bool) -> Self {
        Self {
            vocab,
            do_lower_case,
        }
    }

    /// Tokenize and encode a single text.
    ///
    /// Pipeline: text → pre-tokenize → WordPiece → add [CLS]/[SEP] → truncate → pad
    ///
    /// # Arguments
    /// - `text`: Input text
    /// - `max_length`: Maximum sequence length (including special tokens)
    pub fn encode(&self, text: &str, max_length: usize) -> Result<Encoding> {
        if max_length < 2 {
            return Err(HypEmbedError::Tokenizer(
                "max_length must be at least 2 (for [CLS] and [SEP])".into(),
            ));
        }

        // Step 1: Pre-tokenize
        let words = pre_tokenize::pre_tokenize(text, self.do_lower_case);

        // Step 2: WordPiece tokenize each word
        let mut wp_tokens: Vec<String> = Vec::new();
        for word in &words {
            let subtokens = wordpiece::wordpiece_tokenize(word, &self.vocab);
            wp_tokens.extend(subtokens);
        }

        // Step 3: Truncate to max_length - 2 (leave room for CLS and SEP)
        let max_content = max_length - 2;
        if wp_tokens.len() > max_content {
            wp_tokens.truncate(max_content);
        }

        // Step 4: Build input_ids with [CLS] ... [SEP]
        let mut input_ids = Vec::with_capacity(max_length);
        input_ids.push(self.vocab.cls_id());
        for token in &wp_tokens {
            input_ids.push(self.vocab.token_id(token));
        }
        input_ids.push(self.vocab.sep_id());

        let real_len = input_ids.len();

        // Step 5: Pad to max_length
        let pad_id = self.vocab.pad_id();
        while input_ids.len() < max_length {
            input_ids.push(pad_id);
        }

        // Step 6: Build attention mask and token type IDs
        let mut attention_mask = vec![0u32; max_length];
        for i in 0..real_len {
            attention_mask[i] = 1;
        }

        let token_type_ids = vec![0u32; max_length];

        Ok(Encoding {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    /// Encode a batch of texts in parallel.
    ///
    /// Uses rayon for parallel tokenization. All encodings are padded to `max_length`.
    pub fn encode_batch(&self, texts: &[&str], max_length: usize) -> Result<Vec<Encoding>> {
        texts
            .par_iter()
            .map(|text| self.encode(text, max_length))
            .collect()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Access the vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> Tokenizer {
        let tokens = vec![
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "hello", "world", "un", "##aff", "##able",
            "rust", "##ing", "##s", "the", "a",
            ",", ".", "!", "?",
        ];
        let content = tokens.join("\n");
        let dir = std::env::temp_dir().join("hypembed_test_encode");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vocab.txt");
        std::fs::write(&path, &content).unwrap();
        Tokenizer::new(&path, true).unwrap()
    }

    #[test]
    fn test_encode_basic() {
        let tok = make_test_tokenizer();
        let enc = tok.encode("Hello World", 10).unwrap();
        // [CLS] hello world [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
        assert_eq!(enc.input_ids.len(), 10);
        assert_eq!(enc.input_ids[0], tok.vocab().cls_id()); // [CLS]
        assert_eq!(enc.input_ids[1], tok.vocab().token_id("hello"));
        assert_eq!(enc.input_ids[2], tok.vocab().token_id("world"));
        assert_eq!(enc.input_ids[3], tok.vocab().sep_id()); // [SEP]
        assert_eq!(enc.input_ids[4], tok.vocab().pad_id()); // [PAD]
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 1, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_encode_truncation() {
        let tok = make_test_tokenizer();
        // max_length=4 means 2 content tokens max
        let enc = tok.encode("Hello World Rust", 4).unwrap();
        assert_eq!(enc.input_ids.len(), 4);
        assert_eq!(enc.input_ids[0], tok.vocab().cls_id());
        assert_eq!(enc.input_ids[3], tok.vocab().sep_id());
    }

    #[test]
    fn test_encode_batch() {
        let tok = make_test_tokenizer();
        let batch = tok.encode_batch(&["Hello", "World"], 8).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].input_ids.len(), 8);
        assert_eq!(batch[1].input_ids.len(), 8);
    }

    #[test]
    fn test_attention_mask_correctness() {
        let tok = make_test_tokenizer();
        let enc = tok.encode("Hello", 6).unwrap();
        // [CLS] hello [SEP] [PAD] [PAD] [PAD]
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_token_type_ids() {
        let tok = make_test_tokenizer();
        let enc = tok.encode("Hello", 6).unwrap();
        assert_eq!(enc.token_type_ids, vec![0, 0, 0, 0, 0, 0]);
    }
}
