//! Longest Common Subsequence (LCS) metric.

use crate::metrics::SimilarityMetric;

/// LCS computes the longest common subsequence length between two strings
/// and derives a normalized similarity score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Lcs;

impl SimilarityMetric for Lcs {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        lcs_similarity(a, b)
    }
}

/// Computes the length of the longest common subsequence.
///
/// Uses O(m*n) DP with single-row space optimization.
#[must_use]
pub fn lcs_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 || b_len == 0 {
        return 0;
    }

    let mut prev = vec![0usize; b_len + 1];

    for i in 1..=a_len {
        let mut prev_diag = 0;
        for j in 1..=b_len {
            let old = prev[j];
            if a_chars[i - 1] == b_chars[j - 1] {
                prev[j] = prev_diag + 1;
            } else {
                prev[j] = prev[j].max(prev[j - 1]);
            }
            prev_diag = old;
        }
    }

    prev[b_len]
}

/// Computes normalized LCS similarity: `2 * lcs_len / (len_a + len_b)`.
///
/// Returns 1.0 for two empty strings, 0.0 if one is empty.
#[must_use]
pub fn lcs_similarity(a: &str, b: &str) -> f64 {
    let a_len = a.chars().count();
    let b_len = b.chars().count();
    let total = a_len + b_len;

    if total == 0 {
        return 1.0;
    }

    let lcs = lcs_length(a, b);
    (2 * lcs) as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert_eq!(lcs_length("hello", "hello"), 5);
        assert!(approx_eq(lcs_similarity("hello", "hello"), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert_eq!(lcs_length("", ""), 0);
        assert_eq!(lcs_length("abc", ""), 0);
        assert_eq!(lcs_length("", "abc"), 0);
        assert!(approx_eq(lcs_similarity("", ""), 1.0));
    }

    #[test]
    fn known_values() {
        // "abcde" and "ace" -> LCS is "ace" = 3
        assert_eq!(lcs_length("abcde", "ace"), 3);
        // similarity = 2*3 / (5+3) = 0.75
        assert!(approx_eq(lcs_similarity("abcde", "ace"), 0.75));
    }

    #[test]
    fn symmetry() {
        assert_eq!(lcs_length("abc", "bca"), lcs_length("bca", "abc"));
    }

    #[test]
    fn no_common() {
        assert_eq!(lcs_length("abc", "xyz"), 0);
        assert!(approx_eq(lcs_similarity("abc", "xyz"), 0.0));
    }

    #[test]
    fn unicode() {
        assert_eq!(lcs_length("cafe\u{0301}", "cafe"), 4);
    }
}
