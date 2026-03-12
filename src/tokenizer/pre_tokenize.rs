/// Pre-tokenization: text normalization and splitting.
///
/// Implements BERT-style basic tokenization:
/// 1. Convert to lowercase (for uncased models)
/// 2. Strip accents via Unicode NFD decomposition
/// 3. Handle CJK character splitting
/// 4. Remove control characters
/// 5. Split on whitespace
/// 6. Split on punctuation (each punctuation character becomes its own token)
///
/// This matches the behavior of `BasicTokenizer` in the HuggingFace tokenizers library.

use unicode_normalization::UnicodeNormalization;

/// Pre-tokenize a text string into word-level tokens.
///
/// Applies lowercasing, accent stripping, CJK splitting, whitespace splitting,
/// and punctuation splitting.
pub fn pre_tokenize(text: &str, do_lower_case: bool) -> Vec<String> {
    // Step 1: Clean control characters and normalize whitespace
    let text = clean_text(text);

    // Step 2: Handle CJK characters (add spaces around them)
    let text = tokenize_chinese_chars(&text);

    // Step 3: Lowercase
    let text = if do_lower_case {
        text.to_lowercase()
    } else {
        text
    };

    // Step 4: Strip accents (NFD decomposition + remove combining marks)
    let text = if do_lower_case {
        strip_accents(&text)
    } else {
        text
    };

    // Step 5: Split on whitespace and punctuation
    let mut tokens = Vec::new();
    for word in text.split_whitespace() {
        // Split on punctuation: each punctuation char is its own token
        let mut current = String::new();
        for ch in word.chars() {
            if is_punctuation(ch) {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
    }

    tokens
}

/// Remove control characters and normalize whitespace.
///
/// Replaces \t, \n, \r with spaces. Removes zero-width and other control characters.
fn clean_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for ch in text.chars() {
        // Skip null and replacement characters
        if ch == '\0' || ch == '\u{FFFD}' {
            continue;
        }
        // Skip control characters (except whitespace-like ones)
        if is_control(ch) {
            continue;
        }
        // Normalize whitespace variants to a single space
        if is_whitespace(ch) {
            output.push(' ');
        } else {
            output.push(ch);
        }
    }
    output
}

/// Check if a character is a whitespace character.
///
/// Follows BERT's definition: space, tab, newline, carriage return,
/// plus Unicode whitespace categories.
fn is_whitespace(ch: char) -> bool {
    if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
        return true;
    }
    // Unicode category Zs (space separator)
    matches!(ch,
        '\u{00A0}' | '\u{1680}' | '\u{2000}'..='\u{200A}' |
        '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// Check if a character is a control character.
///
/// Control characters are Unicode category Cc, excluding whitespace chars
/// that we handle separately.
fn is_control(ch: char) -> bool {
    if ch == '\t' || ch == '\n' || ch == '\r' {
        return false; // Handled as whitespace
    }
    let cp = ch as u32;
    // Unicode Cc category: 0x0000-0x001F and 0x007F-0x009F
    // Also filter zero-width characters
    if cp <= 0x001F || (0x007F..=0x009F).contains(&cp) {
        return true;
    }
    // Zero-width characters
    matches!(ch,
        '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}'
    )
}

/// Check if a character is punctuation.
///
/// Uses the same definition as BERT: ASCII punctuation + Unicode punctuation categories.
fn is_punctuation(ch: char) -> bool {
    let cp = ch as u32;
    // ASCII punctuation ranges
    if (33..=47).contains(&cp)      // ! " # $ % & ' ( ) * + , - . /
        || (58..=64).contains(&cp)  // : ; < = > ? @
        || (91..=96).contains(&cp)  // [ \ ] ^ _ `
        || (123..=126).contains(&cp) // { | } ~
    {
        return true;
    }
    // Unicode punctuation: General_Category is P*
    // Use a broader check for common Unicode punctuation
    if ch.is_ascii_punctuation() {
        return true;
    }
    // Check Unicode general categories for punctuation
    // Covers Pc, Pd, Pe, Pf, Pi, Po, Ps
    let cat = unicode_general_category(ch);
    matches!(cat,
        GeneralCategory::Pc | GeneralCategory::Pd | GeneralCategory::Pe |
        GeneralCategory::Pf | GeneralCategory::Pi | GeneralCategory::Po |
        GeneralCategory::Ps
    )
}

/// Strip accents using Unicode NFD decomposition.
///
/// Decomposes characters into base + combining marks, then removes
/// all combining marks (Unicode category Mn/Mc/Me).
fn strip_accents(text: &str) -> String {
    text.nfd()
        .filter(|ch| !is_combining_mark(*ch))
        .collect()
}

/// Check if a character is a Unicode combining mark.
///
/// Covers categories Mn (non-spacing), Mc (spacing combining), Me (enclosing).
fn is_combining_mark(ch: char) -> bool {
    let cp = ch as u32;
    // Combining Diacritical Marks: U+0300–U+036F
    if (0x0300..=0x036F).contains(&cp) {
        return true;
    }
    // Combining Diacritical Marks Extended: U+1AB0–U+1AFF
    if (0x1AB0..=0x1AFF).contains(&cp) {
        return true;
    }
    // Combining Diacritical Marks Supplement: U+1DC0–U+1DFF
    if (0x1DC0..=0x1DFF).contains(&cp) {
        return true;
    }
    // Combining Half Marks: U+FE20–U+FE2F
    if (0xFE20..=0xFE2F).contains(&cp) {
        return true;
    }
    // Combining Diacritical Marks for Symbols: U+20D0–U+20FF
    if (0x20D0..=0x20FF).contains(&cp) {
        return true;
    }
    false
}

/// Add spaces around CJK characters for proper tokenization.
///
/// CJK characters should each be treated as a separate token in BERT.
fn tokenize_chinese_chars(text: &str) -> String {
    let mut output = String::with_capacity(text.len() + text.len() / 4);
    for ch in text.chars() {
        if is_chinese_char(ch) {
            output.push(' ');
            output.push(ch);
            output.push(' ');
        } else {
            output.push(ch);
        }
    }
    output
}

/// Check if a character is a CJK ideograph.
///
/// Covers the main CJK Unified Ideographs block and extensions,
/// matching the BERT BasicTokenizer behavior.
fn is_chinese_char(ch: char) -> bool {
    let cp = ch as u32;
    // CJK Unified Ideographs: U+4E00–U+9FFF
    if (0x4E00..=0x9FFF).contains(&cp) { return true; }
    // CJK Extension A: U+3400–U+4DBF
    if (0x3400..=0x4DBF).contains(&cp) { return true; }
    // CJK Extension B: U+20000–U+2A6DF
    if (0x20000..=0x2A6DF).contains(&cp) { return true; }
    // CJK Extension C: U+2A700–U+2B73F
    if (0x2A700..=0x2B73F).contains(&cp) { return true; }
    // CJK Extension D: U+2B740–U+2B81F
    if (0x2B740..=0x2B81F).contains(&cp) { return true; }
    // CJK Extension E: U+2B820–U+2CEAF
    if (0x2B820..=0x2CEAF).contains(&cp) { return true; }
    // CJK Compatibility Ideographs: U+F900–U+FAFF
    if (0xF900..=0xFAFF).contains(&cp) { return true; }
    // CJK Compatibility Ideographs Supplement: U+2F800–U+2FA1F
    if (0x2F800..=0x2FA1F).contains(&cp) { return true; }
    false
}

/// Simplified Unicode general category detection for punctuation.
#[derive(Debug, PartialEq)]
#[allow(dead_code)]
enum GeneralCategory {
    Pc, // Connector punctuation
    Pd, // Dash punctuation
    Pe, // Close punctuation
    Pf, // Final punctuation
    Pi, // Initial punctuation
    Po, // Other punctuation
    Ps, // Open punctuation
    Other,
}

/// Determine the Unicode general category for punctuation characters.
///
/// This is a simplified check covering the most common Unicode punctuation
/// ranges. It doesn't cover every possible Unicode punctuation character
/// but handles the vast majority seen in real-world text.
fn unicode_general_category(ch: char) -> GeneralCategory {
    let cp = ch as u32;
    match cp {
        // General punctuation U+2000–U+206F
        0x2010..=0x2015 => GeneralCategory::Pd, // Dashes
        0x2018 | 0x201B | 0x201C | 0x201F => GeneralCategory::Pi, // Opening quotes
        0x2019 | 0x201A | 0x201D | 0x201E => GeneralCategory::Pf, // Closing quotes
        0x2020..=0x2027 => GeneralCategory::Po, // Daggers, bullets, etc.
        0x2030..=0x2038 => GeneralCategory::Po,
        0x203B..=0x203E => GeneralCategory::Po,
        0x2039 => GeneralCategory::Pi, // Single left-pointing angle quotation mark
        0x203A => GeneralCategory::Pf, // Single right-pointing angle quotation mark
        // CJK punctuation
        0x3001 | 0x3002 => GeneralCategory::Po, // 、。
        0x3008 | 0x300A | 0x300C | 0x300E | 0x3010 | 0x3014 | 0x3016 | 0x3018 | 0x301A => GeneralCategory::Ps,
        0x3009 | 0x300B | 0x300D | 0x300F | 0x3011 | 0x3015 | 0x3017 | 0x3019 | 0x301B => GeneralCategory::Pe,
        // Fullwidth forms — specific bracket forms before range
        0xFF08 => GeneralCategory::Ps, // （
        0xFF09 => GeneralCategory::Pe, // ）
        0xFF3B => GeneralCategory::Ps, // ［
        0xFF3D => GeneralCategory::Pe, // ］
        0xFF5B => GeneralCategory::Ps, // ｛
        0xFF5D => GeneralCategory::Pe, // ｝
        0xFF01..=0xFF07 | 0xFF0A..=0xFF0F => GeneralCategory::Po, // ！-／ (excluding brackets)
        0xFF1A..=0xFF1B => GeneralCategory::Po, // ：；
        0xFF1F | 0xFF20 => GeneralCategory::Po, // ？＠
        _ => GeneralCategory::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_split() {
        let tokens = pre_tokenize("Hello World", true);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_punctuation_split() {
        let tokens = pre_tokenize("Hello, World!", true);
        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn test_preserve_case() {
        let tokens = pre_tokenize("Hello World", false);
        assert_eq!(tokens, vec!["Hello", "World"]);
    }

    #[test]
    fn test_multiple_spaces() {
        let tokens = pre_tokenize("  hello   world  ", true);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_complex_punctuation() {
        let tokens = pre_tokenize("it's a test-case.", true);
        assert_eq!(tokens, vec!["it", "'", "s", "a", "test", "-", "case", "."]);
    }

    #[test]
    fn test_accent_stripping() {
        let tokens = pre_tokenize("café résumé naïve", true);
        assert_eq!(tokens, vec!["cafe", "resume", "naive"]);
    }

    #[test]
    fn test_accent_stripping_complex() {
        let tokens = pre_tokenize("über Ärger Zürich", true);
        assert_eq!(tokens, vec!["uber", "arger", "zurich"]);
    }

    #[test]
    fn test_chinese_characters() {
        let tokens = pre_tokenize("我爱Rust", true);
        assert_eq!(tokens, vec!["我", "爱", "rust"]);
    }

    #[test]
    fn test_chinese_with_spaces() {
        let tokens = pre_tokenize("你好 世界", true);
        assert_eq!(tokens, vec!["你", "好", "世", "界"]);
    }

    #[test]
    fn test_control_characters_removed() {
        // \x00 (null) and \x01 (SOH) are control chars → removed entirely
        let tokens = pre_tokenize("hello\x00world\x01test", true);
        assert_eq!(tokens, vec!["helloworldtest"]);

        // Tab is treated as whitespace, splitting words
        let tokens = pre_tokenize("hello\tworld", true);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_zero_width_characters() {
        // Zero-width space (U+200B) should be removed
        let tokens = pre_tokenize("hello\u{200B}world", true);
        assert_eq!(tokens, vec!["helloworld"]);
    }

    #[test]
    fn test_unicode_whitespace() {
        // Non-breaking space (U+00A0) should be treated as whitespace
        let tokens = pre_tokenize("hello\u{00A0}world", true);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_mixed_script() {
        let tokens = pre_tokenize("Hello世界, это тест!", true);
        assert_eq!(tokens, vec!["hello", "世", "界", ",", "это", "тест", "!"]);
    }

    #[test]
    fn test_empty_string() {
        let tokens = pre_tokenize("", true);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_replacement_char_removed() {
        let tokens = pre_tokenize("hello\u{FFFD}world", true);
        assert_eq!(tokens, vec!["helloworld"]);
    }
}
