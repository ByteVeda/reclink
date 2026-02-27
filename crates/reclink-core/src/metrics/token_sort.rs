//! Token Sort Ratio — sort tokens alphabetically then compare.

use crate::metrics::levenshtein::Levenshtein;
use crate::metrics::{DistanceMetric, SimilarityMetric};

/// Token Sort Ratio normalizes word order by sorting whitespace-delimited
/// tokens alphabetically before comparing with Levenshtein similarity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokenSort;

impl SimilarityMetric for TokenSort {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        token_sort_ratio(a, b)
    }
}

/// Sorts tokens and joins with a single space.
fn sort_tokens(s: &str) -> String {
    let mut tokens: Vec<&str> = s.split_whitespace().collect();
    tokens.sort_unstable();
    tokens.join(" ")
}

/// Computes the token sort ratio between two strings.
///
/// Tokenizes both strings on whitespace, sorts tokens alphabetically,
/// rejoins with a single space, then computes Levenshtein normalized similarity.
#[must_use]
pub fn token_sort_ratio(a: &str, b: &str) -> f64 {
    let sorted_a = sort_tokens(a);
    let sorted_b = sort_tokens(b);
    Levenshtein
        .normalized_similarity(&sorted_a, &sorted_b)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn reordered_tokens() {
        assert!(approx_eq(token_sort_ratio("John Smith", "Smith John"), 1.0));
    }

    #[test]
    fn identical() {
        assert!(approx_eq(
            token_sort_ratio("hello world", "hello world"),
            1.0
        ));
    }

    #[test]
    fn different_strings() {
        let sim = token_sort_ratio("John Smith", "Jane Doe");
        assert!(sim < 0.5);
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(token_sort_ratio("", ""), 1.0));
        assert!(approx_eq(token_sort_ratio("abc", ""), 0.0));
        assert!(approx_eq(token_sort_ratio("", "abc"), 0.0));
    }

    #[test]
    fn extra_whitespace() {
        assert!(approx_eq(
            token_sort_ratio("  John   Smith  ", "Smith John"),
            1.0
        ));
    }
}
