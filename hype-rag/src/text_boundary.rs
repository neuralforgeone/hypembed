//! Unicode-safe string boundaries for chunking and display truncation.
//!
//! All operations count and slice by Unicode scalar values, never byte indices.

/// Take up to `max_chars` Unicode scalar values from the start of `s`.
pub fn safe_char_prefix(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// Truncate to `max_chars` scalars, appending `"..."` when shortened.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    format!("{}...", safe_char_prefix(s, max_chars))
}

/// Split `s` into overlapping chunks of at most `chunk_chars` scalars each.
pub fn chunk_by_chars(s: &str, chunk_chars: usize, overlap_chars: usize) -> Vec<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let total = trimmed.chars().count();
    if total <= chunk_chars {
        return vec![trimmed.to_string()];
    }

    let step = chunk_chars.saturating_sub(overlap_chars).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < total {
        let piece: String = trimmed.chars().skip(start).take(chunk_chars).collect();
        let piece = piece.trim();
        if !piece.is_empty() {
            chunks.push(piece.to_string());
        }
        if start + chunk_chars >= total {
            break;
        }
        start += step;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_char_prefix_respects_scalar_count() {
        assert_eq!(safe_char_prefix("café", 3), "caf");
        assert_eq!(safe_char_prefix("日本語", 2), "日本");
    }

    #[test]
    fn truncate_chars_does_not_split_multibyte_char() {
        let text = "café résumé";
        let out = truncate_chars(text, 5);
        assert_eq!(out, "café ...");
        assert!(std::str::from_utf8(out.as_bytes()).is_ok());
    }

    #[test]
    fn chunk_by_chars_handles_unicode_boundaries() {
        let text = "日本語".repeat(50);
        let chunks = chunk_by_chars(&text, 20, 5);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 20);
            assert!(std::str::from_utf8(chunk.as_bytes()).is_ok());
        }
    }

    #[test]
    fn chunk_by_chars_cafe_boundary_would_panic_with_byte_slice() {
        let text = format!("{}END", "é".repeat(10));
        let chunks = chunk_by_chars(&text, 5, 2);
        assert!(!chunks.is_empty());
        for chunk in chunks {
            let _ = chunk.as_bytes();
        }
    }
}