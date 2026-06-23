//! Text chunking with fixed size and overlap.

use crate::text_boundary::chunk_by_chars;

/// Default chunk size in characters (~512 tokens).
pub const DEFAULT_CHUNK_CHARS: usize = 1500;
/// Default overlap in characters (~64 tokens).
pub const DEFAULT_OVERLAP_CHARS: usize = 200;

/// A text chunk extracted from a source document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub path: String,
    pub chunk_index: usize,
    pub text: String,
}

/// Split text into overlapping chunks of approximately fixed character length.
pub fn chunk_text(
    path: &str,
    content: &str,
    chunk_chars: usize,
    overlap_chars: usize,
) -> Vec<Chunk> {
    chunk_by_chars(content, chunk_chars, overlap_chars)
        .into_iter()
        .enumerate()
        .map(|(chunk_index, text)| Chunk {
            path: path.to_string(),
            chunk_index,
            text,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_splits_with_overlap() {
        let text = "a".repeat(3000);
        let chunks = chunk_text("doc.md", &text, 1000, 100);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].path, "doc.md");
        assert_eq!(chunks[0].chunk_index, 0);
        assert!(chunks[0].text.chars().count() <= 1000);
    }

    #[test]
    fn chunk_text_empty_returns_none() {
        assert!(chunk_text("x.txt", "   \n", 100, 10).is_empty());
    }

    #[test]
    fn chunk_text_short_is_single_chunk() {
        let chunks = chunk_text("a.txt", "hello world", 1000, 50);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
    }

    #[test]
    fn chunk_text_unicode_safe() {
        let chunks = chunk_text("u.md", "café 日本語 test", 4, 1);
        assert!(!chunks.is_empty());
        for c in chunks {
            assert!(std::str::from_utf8(c.text.as_bytes()).is_ok());
        }
    }
}