//! Levenshtein edit distance using the Wagner-Fischer algorithm with single-row optimization.

use crate::error::Result;
use crate::metrics::DistanceMetric;

/// Levenshtein distance computes the minimum number of single-character edits
/// (insertions, deletions, substitutions) needed to transform one string into another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// Computes Levenshtein distance with early termination.
///
/// Returns `None` if the distance exceeds `max_distance`, `Some(distance)` otherwise.
/// Uses length filtering and row-minimum pruning for O(kn) performance on close matches.
#[must_use]
pub fn levenshtein_distance_threshold(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let (short, long) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };

    let short_len = short.len();
    let long_len = long.len();

    // Length filter: if length difference alone exceeds threshold, bail out
    if long_len - short_len > max_distance {
        return None;
    }

    if short_len == 0 {
        return Some(long_len);
    }

    let mut prev_row: Vec<usize> = (0..=short_len).collect();

    for i in 1..=long_len {
        let mut prev_diag = prev_row[0];
        prev_row[0] = i;
        let mut row_min = prev_row[0];

        for j in 1..=short_len {
            let old_diag = prev_row[j];
            let cost = if long[i - 1] == short[j - 1] { 0 } else { 1 };

            prev_row[j] = (prev_row[j] + 1)
                .min(prev_row[j - 1] + 1)
                .min(prev_diag + cost);

            row_min = row_min.min(prev_row[j]);
            prev_diag = old_diag;
        }

        // If the minimum value in this row exceeds the threshold, abort early
        if row_min > max_distance {
            return None;
        }
    }

    let result = prev_row[short_len];
    if result > max_distance {
        None
    } else {
        Some(result)
    }
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

    #[test]
    fn threshold_within() {
        // kitten -> sitting = 3, threshold 3 should succeed
        assert_eq!(
            levenshtein_distance_threshold("kitten", "sitting", 3),
            Some(3)
        );
        assert_eq!(
            levenshtein_distance_threshold("kitten", "sitting", 5),
            Some(3)
        );
    }

    #[test]
    fn threshold_exceeded() {
        // kitten -> sitting = 3, threshold 2 should return None
        assert_eq!(levenshtein_distance_threshold("kitten", "sitting", 2), None);
        assert_eq!(levenshtein_distance_threshold("kitten", "sitting", 1), None);
    }

    #[test]
    fn threshold_length_filter() {
        // "a" -> "abcdef" = 5, length diff alone is 5 which exceeds threshold 2
        assert_eq!(levenshtein_distance_threshold("a", "abcdef", 2), None);
    }

    #[test]
    fn threshold_empty() {
        assert_eq!(levenshtein_distance_threshold("", "", 0), Some(0));
        assert_eq!(levenshtein_distance_threshold("abc", "", 3), Some(3));
        assert_eq!(levenshtein_distance_threshold("abc", "", 2), None);
    }

    #[test]
    fn threshold_identical() {
        assert_eq!(levenshtein_distance_threshold("hello", "hello", 0), Some(0));
    }
}
