/// WordPiece tokenization algorithm.
///
/// Given a word (already pre-tokenized), splits it into subword tokens
/// using a greedy longest-match-first strategy with `##` prefix for
/// continuation tokens.
///
/// # Algorithm
///
/// For each word:
/// 1. Try to match the longest prefix in the vocabulary
/// 2. If a match is found, consume it and continue with `##` prefix for the remainder
/// 3. If no match is found for any prefix, the entire word maps to [UNK]
///
/// This matches the original BERT WordPiece implementation.

use crate::tokenizer::vocab::Vocab;

/// Maximum subword length to consider (prevents pathological cases).
const MAX_WORD_LEN: usize = 200;

/// Tokenize a single word into WordPiece tokens.
///
/// Returns a vector of subword token strings (e.g., ["un", "##aff", "##able"]).
/// If the word cannot be tokenized, returns ["[UNK]"].
pub fn wordpiece_tokenize(word: &str, vocab: &Vocab) -> Vec<String> {
    if word.is_empty() {
        return vec![];
    }

    // If word is too long, treat as unknown
    if word.len() > MAX_WORD_LEN {
        return vec!["[UNK]".to_string()];
    }

    let chars: Vec<char> = word.chars().collect();
    let mut tokens = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let mut end = chars.len();
        let mut found = false;

        while start < end {
            let substr: String = chars[start..end].iter().collect();
            let candidate = if start > 0 {
                format!("##{}", substr)
            } else {
                substr
            };

            if vocab.contains(&candidate) {
                tokens.push(candidate);
                found = true;
                break;
            }
            end -= 1;
        }

        if !found {
            // No subword match found for any prefix starting at `start`
            return vec!["[UNK]".to_string()];
        }

        start = end;
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a small test vocabulary.
    fn make_test_vocab() -> Vocab {
        let tokens = vec![
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "hello", "world", "un", "##aff", "##able",
            "rust", "##ing", "##s", "the", "a",
        ];
        let content = tokens.join("\n");
        let dir = std::env::temp_dir().join("hypembed_test_vocab");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vocab.txt");
        std::fs::write(&path, &content).unwrap();
        Vocab::load(&path).unwrap()
    }

    #[test]
    fn test_simple_word() {
        let vocab = make_test_vocab();
        assert_eq!(wordpiece_tokenize("hello", &vocab), vec!["hello"]);
    }

    #[test]
    fn test_subword_split() {
        let vocab = make_test_vocab();
        let tokens = wordpiece_tokenize("unaffable", &vocab);
        assert_eq!(tokens, vec!["un", "##aff", "##able"]);
    }

    #[test]
    fn test_unknown_word() {
        let vocab = make_test_vocab();
        let tokens = wordpiece_tokenize("xyzzy", &vocab);
        assert_eq!(tokens, vec!["[UNK]"]);
    }

    #[test]
    fn test_empty() {
        let vocab = make_test_vocab();
        let tokens = wordpiece_tokenize("", &vocab);
        assert!(tokens.is_empty());
    }
}
