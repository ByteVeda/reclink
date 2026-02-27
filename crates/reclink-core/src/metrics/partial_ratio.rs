//! Partial Ratio — best substring match similarity.

use crate::metrics::levenshtein::Levenshtein;
use crate::metrics::{DistanceMetric, SimilarityMetric};

/// Partial Ratio finds the best matching substring of the longer string
/// against the shorter string using Levenshtein similarity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PartialRatio;

impl SimilarityMetric for PartialRatio {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        partial_ratio(a, b)
    }
}

/// Computes the partial ratio between two strings.
///
/// Takes the shorter string and slides it over the longer string, comparing
/// against each substring of matching length. Returns the maximum Levenshtein
/// normalized similarity found.
#[must_use]
pub fn partial_ratio(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.is_empty() && b_chars.is_empty() {
        return 1.0;
    }
    if a_chars.is_empty() || b_chars.is_empty() {
        return 0.0;
    }

    let (short, long) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };

    let short_len = short.len();
    let long_len = long.len();

    if short_len == long_len {
        return Levenshtein.normalized_similarity(a, b).unwrap_or(0.0);
    }

    let short_str: String = short.iter().collect();
    let mut best = 0.0f64;

    for i in 0..=(long_len - short_len) {
        let substr: String = long[i..i + short_len].iter().collect();
        let sim = Levenshtein
            .normalized_similarity(&short_str, &substr)
            .unwrap_or(0.0);
        best = best.max(sim);
        if best >= 1.0 {
            return 1.0;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn substring_match() {
        assert!(approx_eq(partial_ratio("test", "this is a test"), 1.0));
    }

    #[test]
    fn identical() {
        assert!(approx_eq(partial_ratio("hello", "hello"), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(partial_ratio("", ""), 1.0));
        assert!(approx_eq(partial_ratio("abc", ""), 0.0));
        assert!(approx_eq(partial_ratio("", "abc"), 0.0));
    }

    #[test]
    fn no_match() {
        let sim = partial_ratio("xyz", "abc");
        assert!(sim < 0.5);
    }

    #[test]
    fn shorter_is_first_arg() {
        // Symmetry: doesn't matter which arg is shorter
        let a = partial_ratio("test", "this is a test string");
        let b = partial_ratio("this is a test string", "test");
        assert!(approx_eq(a, b));
    }
}
