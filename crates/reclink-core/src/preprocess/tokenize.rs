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
}
