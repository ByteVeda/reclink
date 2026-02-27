//! Damerau-Levenshtein distance (optimal string alignment variant).
//!
//! Extends Levenshtein with transposition of adjacent characters.

use crate::error::Result;
use crate::metrics::DistanceMetric;

/// Damerau-Levenshtein distance counts insertions, deletions, substitutions,
/// and transpositions of adjacent characters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DamerauLevenshtein;

impl DistanceMetric for DamerauLevenshtein {
    fn distance(&self, a: &str, b: &str) -> Result<usize> {
        Ok(damerau_levenshtein_distance(a, b))
    }
}

/// Computes the optimal string alignment distance (restricted Damerau-Levenshtein).
///
/// Time: O(m*n), Space: O(m*n)
#[must_use]
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut matrix = vec![vec![0usize; b_len + 1]; a_len + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(a_len + 1) {
        row[0] = i;
    }
    for (j, val) in matrix[0].iter_mut().enumerate().take(b_len + 1) {
        *val = j;
    }

    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);

            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                matrix[i][j] = matrix[i][j].min(matrix[i - 2][j - 2] + 1);
            }
        }
    }

    matrix[a_len][b_len]
}

/// Computes Damerau-Levenshtein distance with early termination.
///
/// Returns `None` if the distance exceeds `max_distance`, `Some(distance)` otherwise.
#[must_use]
pub fn damerau_levenshtein_distance_threshold(
    a: &str,
    b: &str,
    max_distance: usize,
) -> Option<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    // Length filter
    if a_len.abs_diff(b_len) > max_distance {
        return None;
    }

    if a_len == 0 {
        return if b_len > max_distance {
            None
        } else {
            Some(b_len)
        };
    }
    if b_len == 0 {
        return if a_len > max_distance {
            None
        } else {
            Some(a_len)
        };
    }

    let mut matrix = vec![vec![0usize; b_len + 1]; a_len + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(a_len + 1) {
        row[0] = i;
    }
    for (j, val) in matrix[0].iter_mut().enumerate().take(b_len + 1) {
        *val = j;
    }

    for i in 1..=a_len {
        let mut row_min = usize::MAX;

        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);

            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                matrix[i][j] = matrix[i][j].min(matrix[i - 2][j - 2] + 1);
            }

            row_min = row_min.min(matrix[i][j]);
        }

        if row_min > max_distance {
            return None;
        }
    }

    let result = matrix[a_len][b_len];
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
        assert_eq!(damerau_levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn empty_strings() {
        assert_eq!(damerau_levenshtein_distance("", ""), 0);
        assert_eq!(damerau_levenshtein_distance("abc", ""), 3);
        assert_eq!(damerau_levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn transposition() {
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
        assert_eq!(damerau_levenshtein_distance("abc", "bac"), 1);
    }

    #[test]
    fn known_values() {
        assert_eq!(damerau_levenshtein_distance("ca", "abc"), 3);
    }

    #[test]
    fn symmetry() {
        assert_eq!(
            damerau_levenshtein_distance("abc", "bca"),
            damerau_levenshtein_distance("bca", "abc")
        );
    }

    #[test]
    fn unicode() {
        assert_eq!(damerau_levenshtein_distance("über", "uebr"), 2);
    }

    #[test]
    fn threshold_within() {
        // "ab" -> "ba" = 1 (transposition), threshold 1 should succeed
        assert_eq!(
            damerau_levenshtein_distance_threshold("ab", "ba", 1),
            Some(1)
        );
        assert_eq!(
            damerau_levenshtein_distance_threshold("ab", "ba", 5),
            Some(1)
        );
    }

    #[test]
    fn threshold_exceeded() {
        assert_eq!(damerau_levenshtein_distance_threshold("ca", "abc", 2), None);
    }

    #[test]
    fn threshold_length_filter() {
        assert_eq!(
            damerau_levenshtein_distance_threshold("a", "abcdef", 2),
            None
        );
    }

    #[test]
    fn threshold_empty() {
        assert_eq!(damerau_levenshtein_distance_threshold("", "", 0), Some(0));
        assert_eq!(
            damerau_levenshtein_distance_threshold("abc", "", 3),
            Some(3)
        );
        assert_eq!(damerau_levenshtein_distance_threshold("abc", "", 2), None);
    }

    #[test]
    fn threshold_identical() {
        assert_eq!(
            damerau_levenshtein_distance_threshold("hello", "hello", 0),
            Some(0)
        );
    }
}
