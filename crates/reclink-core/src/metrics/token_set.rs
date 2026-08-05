//! Token Set Ratio — set-based fuzzy matching.

use std::collections::BTreeSet;

use crate::metrics::levenshtein::Levenshtein;
use crate::metrics::{DistanceMetric, SimilarityMetric};

/// Token Set Ratio computes similarity by splitting tokens into intersection
/// and remainder sets, then comparing the most favorable combination.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokenSet;

impl SimilarityMetric for TokenSet {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        token_set_ratio(a, b)
    }
}

/// Computes the token set ratio between two strings.
///
/// Tokenizes both strings, computes intersection and remainder sets, then
/// returns the maximum Levenshtein similarity among:
/// - sorted intersection vs combined_a (intersection + remainder_a)
/// - sorted intersection vs combined_b (intersection + remainder_b)
/// - combined_a vs combined_b
#[must_use]
pub fn token_set_ratio(a: &str, b: &str) -> f64 {
    let owned_a = crate::preprocess::tokenize::tokenize_for_matching(a);
    let owned_b = crate::preprocess::tokenize::tokenize_for_matching(b);
    let tokens_a: BTreeSet<&str> = owned_a.iter().map(|s| s.as_str()).collect();
    let tokens_b: BTreeSet<&str> = owned_b.iter().map(|s| s.as_str()).collect();

    if tokens_a.is_empty() && tokens_b.is_empty() {
        return 1.0;
    }
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }

    let intersection: BTreeSet<&str> = tokens_a.intersection(&tokens_b).copied().collect();
    let remainder_a: BTreeSet<&str> = tokens_a.difference(&tokens_b).copied().collect();
    let remainder_b: BTreeSet<&str> = tokens_b.difference(&tokens_a).copied().collect();

    let sorted_inter: String = intersection.iter().copied().collect::<Vec<_>>().join(" ");

    let combined_a = if remainder_a.is_empty() {
        sorted_inter.clone()
    } else {
        let rem: String = remainder_a.iter().copied().collect::<Vec<_>>().join(" ");
        if sorted_inter.is_empty() {
            rem
        } else {
            format!("{sorted_inter} {rem}")
        }
    };

    let combined_b = if remainder_b.is_empty() {
        sorted_inter.clone()
    } else {
        let rem: String = remainder_b.iter().copied().collect::<Vec<_>>().join(" ");
        if sorted_inter.is_empty() {
            rem
        } else {
            format!("{sorted_inter} {rem}")
        }
    };

    let lev = Levenshtein;
    let score1 = lev
        .normalized_similarity(&sorted_inter, &combined_a)
        .unwrap_or(0.0);
    let score2 = lev
        .normalized_similarity(&sorted_inter, &combined_b)
        .unwrap_or(0.0);
    let score3 = lev
        .normalized_similarity(&combined_a, &combined_b)
        .unwrap_or(0.0);

    score1.max(score2).max(score3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(
            token_set_ratio("hello world", "hello world"),
            1.0
        ));
    }

    #[test]
    fn subset_match() {
        // "New York" is a subset of "New York City" -> high score
        let sim = token_set_ratio("New York", "New York City");
        assert!(sim > 0.8);
    }

    #[test]
    fn partial_overlap() {
        let sim = token_set_ratio("cat dog bird", "cat fish bird");
        assert!(sim > 0.5);
    }

    #[test]
    fn no_overlap() {
        let sim = token_set_ratio("abc def", "xyz uvw");
        assert!(sim < 0.5);
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(token_set_ratio("", ""), 1.0));
        assert!(approx_eq(token_set_ratio("abc", ""), 0.0));
        assert!(approx_eq(token_set_ratio("", "abc"), 0.0));
    }
}
