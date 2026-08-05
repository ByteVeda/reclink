//! Weighted edit distance with configurable operation costs.

use crate::metrics::SimilarityMetric;

/// Weighted Levenshtein computes edit distance with configurable costs for
/// insert, delete, substitute, and transpose operations.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightedLevenshtein {
    /// Cost of inserting a character.
    pub insert_cost: f64,
    /// Cost of deleting a character.
    pub delete_cost: f64,
    /// Cost of substituting a character.
    pub substitute_cost: f64,
    /// Cost of transposing two adjacent characters.
    pub transpose_cost: f64,
}

impl Default for WeightedLevenshtein {
    fn default() -> Self {
        Self {
            insert_cost: 1.0,
            delete_cost: 1.0,
            substitute_cost: 1.0,
            transpose_cost: 1.0,
        }
    }
}

impl SimilarityMetric for WeightedLevenshtein {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        let dist = weighted_levenshtein_distance(
            a,
            b,
            self.insert_cost,
            self.delete_cost,
            self.substitute_cost,
            self.transpose_cost,
        );
        let a_len = a.chars().count();
        let b_len = b.chars().count();
        let max_len = a_len.max(b_len);
        if max_len == 0 {
            return 1.0;
        }
        let max_cost = self.insert_cost.max(self.delete_cost);
        let max_possible = max_len as f64 * max_cost;
        if max_possible <= 0.0 {
            return 1.0;
        }
        (1.0 - dist / max_possible).max(0.0)
    }
}

/// Computes weighted edit distance with configurable operation costs.
///
/// Uses a modified Wagner-Fischer DP that supports insert, delete, substitute,
/// and transpose operations with independent float costs.
///
/// Time: O(m*n), Space: O(m*n)
#[must_use]
pub fn weighted_levenshtein_distance(
    a: &str,
    b: &str,
    insert_cost: f64,
    delete_cost: f64,
    substitute_cost: f64,
    transpose_cost: f64,
) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len as f64 * insert_cost;
    }
    if b_len == 0 {
        return a_len as f64 * delete_cost;
    }

    // Full matrix needed for transpose lookback
    let mut dp = vec![vec![0.0; b_len + 1]; a_len + 1];

    for (i, row) in dp.iter_mut().enumerate().take(a_len + 1) {
        row[0] = i as f64 * delete_cost;
    }
    for (j, val) in dp[0].iter_mut().enumerate().take(b_len + 1) {
        *val = j as f64 * insert_cost;
    }

    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                dp[i - 1][j - 1]
            } else {
                dp[i - 1][j - 1] + substitute_cost
            };

            dp[i][j] = cost
                .min(dp[i - 1][j] + delete_cost)
                .min(dp[i][j - 1] + insert_cost);

            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                dp[i][j] = dp[i][j].min(dp[i - 2][j - 2] + transpose_cost);
            }
        }
    }

    dp[a_len][b_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn identical_strings() {
        assert!(approx_eq(
            weighted_levenshtein_distance("hello", "hello", 1.0, 1.0, 1.0, 1.0),
            0.0
        ));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(
            weighted_levenshtein_distance("", "", 1.0, 1.0, 1.0, 1.0),
            0.0
        ));
        assert!(approx_eq(
            weighted_levenshtein_distance("abc", "", 1.0, 1.0, 1.0, 1.0),
            3.0
        ));
        assert!(approx_eq(
            weighted_levenshtein_distance("", "abc", 1.0, 1.0, 1.0, 1.0),
            3.0
        ));
    }

    #[test]
    fn uniform_costs_match_levenshtein() {
        // With uniform costs of 1.0, should behave like regular Levenshtein
        assert!(approx_eq(
            weighted_levenshtein_distance("kitten", "sitting", 1.0, 1.0, 1.0, 1.0),
            3.0
        ));
    }

    #[test]
    fn asymmetric_costs() {
        // Insert costs 2, so inserting 3 chars into "" -> "abc" costs 6
        assert!(approx_eq(
            weighted_levenshtein_distance("", "abc", 2.0, 1.0, 1.0, 1.0),
            6.0
        ));
        // Delete costs 3, so deleting 2 chars from "ab" -> "" costs 6
        assert!(approx_eq(
            weighted_levenshtein_distance("ab", "", 1.0, 3.0, 1.0, 1.0),
            6.0
        ));
    }

    #[test]
    fn transpose() {
        // "ab" -> "ba" with transpose cost 0.5
        assert!(approx_eq(
            weighted_levenshtein_distance("ab", "ba", 1.0, 1.0, 1.0, 0.5),
            0.5
        ));
    }

    #[test]
    fn similarity_identical() {
        let m = WeightedLevenshtein::default();
        assert!(approx_eq(m.similarity("hello", "hello"), 1.0));
    }

    #[test]
    fn similarity_empty() {
        let m = WeightedLevenshtein::default();
        assert!(approx_eq(m.similarity("", ""), 1.0));
    }
}
