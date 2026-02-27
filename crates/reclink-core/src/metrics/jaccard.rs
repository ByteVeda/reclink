//! Jaccard similarity on whitespace-separated tokens.

use ahash::AHashSet;

use crate::metrics::SimilarityMetric;

/// Jaccard similarity computes the size of the intersection divided by
/// the size of the union of two token sets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Jaccard;

impl SimilarityMetric for Jaccard {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        jaccard_similarity(a, b)
    }
}

/// Computes Jaccard similarity between whitespace-tokenized sets of two strings.
#[must_use]
pub fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let set_a: AHashSet<&str> = a.split_whitespace().collect();
    let set_b: AHashSet<&str> = b.split_whitespace().collect();

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 1.0;
    }

    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn identical() {
        assert!(approx_eq(
            jaccard_similarity("hello world", "hello world"),
            1.0
        ));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(jaccard_similarity("", ""), 1.0));
        assert!(approx_eq(jaccard_similarity("hello", ""), 0.0));
    }

    #[test]
    fn no_overlap() {
        assert!(approx_eq(jaccard_similarity("cat dog", "fish bird"), 0.0));
    }

    #[test]
    fn partial_overlap() {
        assert!(approx_eq(
            jaccard_similarity("cat dog", "cat bird"),
            1.0 / 3.0
        ));
    }

    #[test]
    fn symmetry() {
        let a = jaccard_similarity("hello world", "world foo");
        let b = jaccard_similarity("world foo", "hello world");
        assert!(approx_eq(a, b));
    }
}
