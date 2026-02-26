//! Jaro-Winkler similarity metric.

use crate::metrics::jaro::jaro_similarity;
use crate::metrics::SimilarityMetric;

/// Jaro-Winkler similarity extends Jaro with a prefix bonus that gives
/// higher scores to strings sharing a common prefix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JaroWinkler {
    /// Scaling factor for common prefix bonus. Default: 0.1. Must be <= 0.25.
    pub prefix_weight: f64,
}

impl Default for JaroWinkler {
    fn default() -> Self {
        Self { prefix_weight: 0.1 }
    }
}

impl SimilarityMetric for JaroWinkler {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        jaro_winkler_similarity(a, b, self.prefix_weight)
    }
}

/// Computes Jaro-Winkler similarity with a given prefix weight.
///
/// The prefix weight should be in \[0, 0.25\]. The common prefix length
/// is capped at 4 characters.
#[must_use]
pub fn jaro_winkler_similarity(a: &str, b: &str, prefix_weight: f64) -> f64 {
    let jaro = jaro_similarity(a, b);

    let prefix_len = a
        .chars()
        .zip(b.chars())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();

    jaro + (prefix_len as f64 * prefix_weight * (1.0 - jaro))
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
            jaro_winkler_similarity("hello", "hello", 0.1),
            1.0
        ));
    }

    #[test]
    fn known_values() {
        assert!(approx_eq(
            jaro_winkler_similarity("martha", "marhta", 0.1),
            0.9611
        ));
        assert!(approx_eq(
            jaro_winkler_similarity("dwayne", "duane", 0.1),
            0.84
        ));
        assert!(approx_eq(
            jaro_winkler_similarity("dixon", "dicksonx", 0.1),
            0.8133
        ));
    }

    #[test]
    fn prefix_boost() {
        let jw = jaro_winkler_similarity("abc", "abx", 0.1);
        let j = jaro_similarity("abc", "abx");
        assert!(jw >= j);
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(jaro_winkler_similarity("", "", 0.1), 1.0));
    }

    #[test]
    fn symmetry() {
        let a = jaro_winkler_similarity("abc", "bac", 0.1);
        let b = jaro_winkler_similarity("bac", "abc", 0.1);
        assert!(approx_eq(a, b));
    }
}
