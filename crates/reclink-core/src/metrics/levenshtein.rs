//! Levenshtein edit distance using the Wagner-Fischer algorithm with single-row optimization.

use crate::error::Result;
use crate::metrics::DistanceMetric;

/// Levenshtein distance computes the minimum number of single-character edits
/// (insertions, deletions, substitutions) needed to transform one string into another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Levenshtein;

impl DistanceMetric for Levenshtein {
    fn distance(&self, a: &str, b: &str) -> Result<usize> {
        Ok(levenshtein_distance(a, b))
    }
}

/// Computes Levenshtein distance using single-row Wagner-Fischer optimization.
///
/// Time: O(m*n), Space: O(min(m,n))
#[must_use]
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let (short, long) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };

    let short_len = short.len();
    let long_len = long.len();

    if short_len == 0 {
        return long_len;
    }

    let mut prev_row: Vec<usize> = (0..=short_len).collect();

    for i in 1..=long_len {
        let mut prev_diag = prev_row[0];
        prev_row[0] = i;

        for j in 1..=short_len {
            let old_diag = prev_row[j];
            let cost = if long[i - 1] == short[j - 1] { 0 } else { 1 };

            prev_row[j] = (prev_row[j] + 1)
                .min(prev_row[j - 1] + 1)
                .min(prev_diag + cost);

            prev_diag = old_diag;
        }
    }

    prev_row[short_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn empty_strings() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn known_values() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(levenshtein_distance("flaw", "lawn"), 2);
    }

    #[test]
    fn symmetry() {
        assert_eq!(
            levenshtein_distance("abc", "def"),
            levenshtein_distance("def", "abc")
        );
    }

    #[test]
    fn unicode() {
        assert_eq!(levenshtein_distance("café", "cafe"), 1);
        assert_eq!(levenshtein_distance("日本語", "日本人"), 1);
    }

    #[test]
    fn normalized_similarity() {
        let m = Levenshtein;
        let sim = m.normalized_similarity("kitten", "sitting").unwrap();
        assert!((sim - (1.0 - 3.0 / 7.0)).abs() < 1e-10);
    }
}
