//! Longest Common Substring metric.
//!
//! Uses O(m*n) DP with single-row space optimization: O(min(m,n)) space.

use crate::metrics::SimilarityMetric;

/// Longest Common Substring finds the longest contiguous matching sequence
/// between two strings and derives a normalized similarity score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LongestCommonSubstring;

impl SimilarityMetric for LongestCommonSubstring {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        longest_common_substring_similarity(a, b)
    }
}

/// Computes the length of the longest common substring.
///
/// Uses O(m*n) DP with single-row space optimization.
#[must_use]
pub fn longest_common_substring_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 || b_len == 0 {
        return 0;
    }

    // Use shorter string as column dimension for less memory
    let (row_chars, col_chars) = if a_len >= b_len {
        (&a_chars[..], &b_chars[..])
    } else {
        (&b_chars[..], &a_chars[..])
    };

    let cols = col_chars.len();
    let mut prev = vec![0usize; cols + 1];
    let mut max_len = 0;

    for rc in row_chars {
        let mut prev_diag = 0;
        for j in 1..=cols {
            let old = prev[j];
            if *rc == col_chars[j - 1] {
                prev[j] = prev_diag + 1;
                max_len = max_len.max(prev[j]);
            } else {
                prev[j] = 0;
            }
            prev_diag = old;
        }
    }

    max_len
}

/// Computes normalized longest common substring similarity:
/// `2 * substr_len / (len_a + len_b)`.
///
/// Returns 1.0 for two empty strings, 0.0 if one is empty.
#[must_use]
pub fn longest_common_substring_similarity(a: &str, b: &str) -> f64 {
    let a_len = a.chars().count();
    let b_len = b.chars().count();
    let total = a_len + b_len;

    if total == 0 {
        return 1.0;
    }

    let lcs = longest_common_substring_length(a, b);
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
        assert_eq!(longest_common_substring_length("hello", "hello"), 5);
        assert!(approx_eq(
            longest_common_substring_similarity("hello", "hello"),
            1.0
        ));
    }

    #[test]
    fn empty_strings() {
        assert_eq!(longest_common_substring_length("", ""), 0);
        assert_eq!(longest_common_substring_length("abc", ""), 0);
        assert!(approx_eq(longest_common_substring_similarity("", ""), 1.0));
    }

    #[test]
    fn known_values() {
        // "abcxyz" and "xyzabc" -> longest common substring is "abc" or "xyz" = 3
        assert_eq!(longest_common_substring_length("abcxyz", "xyzabc"), 3);
        // similarity = 2*3 / (6+6) = 0.5
        assert!(approx_eq(
            longest_common_substring_similarity("abcxyz", "xyzabc"),
            0.5
        ));
    }

    #[test]
    fn no_common() {
        assert_eq!(longest_common_substring_length("abc", "xyz"), 0);
        assert!(approx_eq(
            longest_common_substring_similarity("abc", "xyz"),
            0.0
        ));
    }

    #[test]
    fn single_char_match() {
        assert_eq!(longest_common_substring_length("axb", "ayb"), 1);
    }

    #[test]
    fn symmetry() {
        assert_eq!(
            longest_common_substring_length("abc", "bca"),
            longest_common_substring_length("bca", "abc")
        );
    }
}
