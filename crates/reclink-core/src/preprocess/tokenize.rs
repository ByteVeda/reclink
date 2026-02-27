//! Tokenization utilities.

/// Splits a string on whitespace boundaries.
#[must_use]
pub fn whitespace_tokenize(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

/// Generates character n-grams from a string.
#[must_use]
pub fn ngram_tokenize(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n || n == 0 {
        return Vec::new();
    }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

/// Returns `true` if a character is in a CJK code block.
///
/// Covers CJK Unified Ideographs, Extension A, Compatibility Ideographs,
/// Katakana, Hiragana, and Hangul Syllables.
#[inline]
#[must_use]
pub fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
    )
}

/// Tokenizes a string into individual characters.
///
/// Every character becomes its own token. Useful for CJK text where
/// whitespace delimiters are absent.
#[must_use]
pub fn character_tokenize(s: &str) -> Vec<String> {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_string())
        .collect()
}

/// Smart tokenizer that auto-detects CJK vs Latin runs.
///
/// CJK characters become individual tokens. Contiguous Latin/other runs
/// are kept together and split on whitespace.
#[must_use]
pub fn smart_tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut latin_buf = String::new();

    for ch in s.chars() {
        if is_cjk(ch) {
            // Flush any pending Latin buffer
            if !latin_buf.is_empty() {
                for tok in latin_buf.split_whitespace() {
                    tokens.push(tok.to_string());
                }
                latin_buf.clear();
            }
            tokens.push(ch.to_string());
        } else {
            latin_buf.push(ch);
        }
    }

    // Flush remaining Latin text
    if !latin_buf.is_empty() {
        for tok in latin_buf.split_whitespace() {
            tokens.push(tok.to_string());
        }
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace() {
        assert_eq!(
            whitespace_tokenize("hello world foo"),
            vec!["hello", "world", "foo"]
        );
    }

    #[test]
    fn whitespace_empty() {
        assert!(whitespace_tokenize("").is_empty());
    }

    #[test]
    fn ngrams() {
        assert_eq!(ngram_tokenize("hello", 2), vec!["he", "el", "ll", "lo"]);
    }

    #[test]
    fn ngrams_too_short() {
        assert!(ngram_tokenize("a", 2).is_empty());
    }

    #[test]
    fn ngrams_zero() {
        assert!(ngram_tokenize("hello", 0).is_empty());
    }

    #[test]
    fn is_cjk_chinese() {
        assert!(is_cjk('你'));
        assert!(is_cjk('好'));
        assert!(is_cjk('世'));
    }

    #[test]
    fn is_cjk_japanese() {
        assert!(is_cjk('東')); // kanji
        assert!(is_cjk('タ')); // katakana
        assert!(is_cjk('あ')); // hiragana
    }

    #[test]
    fn is_cjk_korean() {
        assert!(is_cjk('한'));
        assert!(is_cjk('글'));
    }

    #[test]
    fn is_cjk_latin() {
        assert!(!is_cjk('a'));
        assert!(!is_cjk('Z'));
        assert!(!is_cjk(' '));
    }

    #[test]
    fn character_tokenize_basic() {
        assert_eq!(
            character_tokenize("東京タワー"),
            vec!["東", "京", "タ", "ワ", "ー"]
        );
    }

    #[test]
    fn character_tokenize_strips_whitespace() {
        assert_eq!(character_tokenize("a b c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn character_tokenize_empty() {
        assert!(character_tokenize("").is_empty());
    }

    #[test]
    fn smart_tokenize_chinese() {
        assert_eq!(smart_tokenize("你好世界"), vec!["你", "好", "世", "界"]);
    }

    #[test]
    fn smart_tokenize_mixed() {
        assert_eq!(smart_tokenize("Hello 世界"), vec!["Hello", "世", "界"]);
    }

    #[test]
    fn smart_tokenize_latin_only() {
        assert_eq!(smart_tokenize("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn smart_tokenize_japanese() {
        let tokens = smart_tokenize("東京タワー");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], "東");
    }

    #[test]
    fn smart_tokenize_empty() {
        assert!(smart_tokenize("").is_empty());
    }

    #[test]
    fn smart_tokenize_mixed_with_spaces() {
        assert_eq!(
            smart_tokenize("Tokyo 東京 is great"),
            vec!["Tokyo", "東", "京", "is", "great"]
        );
    }
}
