//! Damerau-Levenshtein distance (optimal string alignment variant).
//!
//! Extends Levenshtein with transposition of adjacent characters.
//! Uses 3-row rolling DP: O(m·n) time, O(min(m,n)) space.

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

/// 3-row rolling DP for optimal string alignment distance.
///
/// Uses only 3 rows (prev-prev, prev, current) instead of the full matrix,
/// reducing space from O(m·n) to O(min(m,n)).
fn osa_three_row(row_chars: &[char], col_chars: &[char]) -> usize {
    use crate::metrics::scratch::DL_SCRATCH;

    let rows = row_chars.len();
    let cols = col_chars.len();

    DL_SCRATCH.with_borrow_mut(|scratch| {
        scratch.reset(cols);

        for i in 1..=rows {
            scratch.curr[0] = i;

            for j in 1..=cols {
                let cost = if row_chars[i - 1] == col_chars[j - 1] {
                    0
                } else {
                    1
                };

                scratch.curr[j] = (scratch.prev[j] + 1)
                    .min(scratch.curr[j - 1] + 1)
                    .min(scratch.prev[j - 1] + cost);

                if i > 1
                    && j > 1
                    && row_chars[i - 1] == col_chars[j - 2]
                    && row_chars[i - 2] == col_chars[j - 1]
                {
                    scratch.curr[j] = scratch.curr[j].min(scratch.prev_prev[j - 2] + 1);
                }
            }

            // Rotate rows
            std::mem::swap(&mut scratch.prev_prev, &mut scratch.prev);
            std::mem::swap(&mut scratch.prev, &mut scratch.curr);
            for v in scratch.curr.iter_mut() {
                *v = 0;
            }
        }

        scratch.prev[cols]
    })
}

/// Computes the optimal string alignment distance (restricted Damerau-Levenshtein).
///
/// Time: O(m*n), Space: O(min(m,n))
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

    // Use shorter string as columns for less memory
    if a_len <= b_len {
        osa_three_row(&b_chars, &a_chars)
    } else {
        osa_three_row(&a_chars, &b_chars)
    }
}

/// 3-row rolling DP with row-minimum pruning for early termination.
fn osa_three_row_threshold(
    row_chars: &[char],
    col_chars: &[char],
    max_distance: usize,
) -> Option<usize> {
    let rows = row_chars.len();
    let cols = col_chars.len();

    let mut prev_prev = vec![0usize; cols + 1];
    let mut prev: Vec<usize> = (0..=cols).collect();
    let mut curr = vec![0usize; cols + 1];

    for i in 1..=rows {
        curr[0] = i;
        let mut row_min = curr[0];

        for j in 1..=cols {
            let cost = if row_chars[i - 1] == col_chars[j - 1] {
                0
            } else {
                1
            };

            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);

            if i > 1
                && j > 1
                && row_chars[i - 1] == col_chars[j - 2]
                && row_chars[i - 2] == col_chars[j - 1]
            {
                curr[j] = curr[j].min(prev_prev[j - 2] + 1);
            }

            row_min = row_min.min(curr[j]);
        }

        if row_min > max_distance {
            return None;
        }

        std::mem::swap(&mut prev_prev, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
        for v in curr.iter_mut() {
            *v = 0;
        }
    }

    let result = prev[cols];
    if result > max_distance {
        None
    } else {
        Some(result)
    }
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

    // Use shorter string as columns for less memory
    if a_len <= b_len {
        osa_three_row_threshold(&b_chars, &a_chars, max_distance)
    } else {
        osa_three_row_threshold(&a_chars, &b_chars, max_distance)
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

    #[test]
    fn long_strings() {
        let a: String = (0..100).map(|i| (b'a' + (i % 26)) as char).collect();
        let b: String = a
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 50 { 'Z' } else { c })
            .collect();
        assert_eq!(damerau_levenshtein_distance(&a, &b), 1);
    }

    #[test]
    fn transposition_long() {
        let mut a_chars: Vec<char> = (0..100).map(|i| (b'a' + (i % 26)) as char).collect();
        let b_chars = a_chars.clone();
        a_chars.swap(50, 51);
        let a: String = a_chars.into_iter().collect();
        let b: String = b_chars.into_iter().collect();
        assert_eq!(damerau_levenshtein_distance(&a, &b), 1);
    }
}
