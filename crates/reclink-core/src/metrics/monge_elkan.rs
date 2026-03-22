//! Monge-Elkan hybrid token-based similarity metric.
//!
//! For each token in string A, finds the best-matching token in string B
//! using a configurable inner metric. The final score is the average of
//! these best matches.

use crate::metrics::{Metric, SimilarityMetric};

/// Monge-Elkan token-based similarity metric.
///
/// Splits both strings into tokens and, for each token in A, finds the
/// maximum similarity with any token in B using the inner metric.
/// The score is the average of these maximum similarities.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MongeElkan {
    /// The inner metric used to compare individual tokens.
    pub inner_metric: Box<Metric>,
}

impl Default for MongeElkan {
    fn default() -> Self {
        Self {
            inner_metric: Box::new(Metric::default()),
        }
    }
}

impl SimilarityMetric for MongeElkan {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        monge_elkan_similarity(a, b, &self.inner_metric)
    }
}

/// Computes Monge-Elkan similarity between two strings.
///
/// Splits on whitespace, computes max inner similarity for each token in A
/// against all tokens in B, returns the average.
#[must_use]
pub fn monge_elkan_similarity(a: &str, b: &str, inner: &Metric) -> f64 {
    let a_tokens: Vec<&str> = a.split_whitespace().collect();
    let b_tokens: Vec<&str> = b.split_whitespace().collect();

    if a_tokens.is_empty() && b_tokens.is_empty() {
        return 1.0;
    }
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return 0.0;
    }

    let sum: f64 = a_tokens
        .iter()
        .map(|ta| {
            b_tokens
                .iter()
                .map(|tb| inner.similarity(ta, tb))
                .fold(0.0f64, f64::max)
        })
        .sum();

    sum / a_tokens.len() as f64
}

/// Computes Monge-Elkan similarity using the default inner metric (Jaro-Winkler).
#[must_use]
pub fn monge_elkan_default(a: &str, b: &str) -> f64 {
    MongeElkan::default().similarity(a, b)
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
            monge_elkan_default("hello world", "hello world"),
            1.0
        ));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(monge_elkan_default("", ""), 1.0));
        assert!(approx_eq(monge_elkan_default("abc", ""), 0.0));
        assert!(approx_eq(monge_elkan_default("", "abc"), 0.0));
    }

    #[test]
    fn single_token() {
        assert!(approx_eq(monge_elkan_default("john", "john"), 1.0));
    }

    #[test]
    fn token_reordering() {
        // "john smith" vs "smith john" — each token finds its perfect match
        let sim = monge_elkan_default("john smith", "smith john");
        assert!(approx_eq(sim, 1.0));
    }

    #[test]
    fn partial_token_match() {
        // Tokens partially match via inner metric
        let sim = monge_elkan_default("john smith", "jon smyth");
        assert!(sim > 0.5 && sim < 1.0);
    }

    #[test]
    fn completely_different() {
        let sim = monge_elkan_default("abc def", "xyz uvw");
        assert!(sim < 0.5);
    }

    #[test]
    fn asymmetric() {
        // Monge-Elkan is asymmetric: avg over A's tokens vs avg over B's tokens
        let ab = monge_elkan_default("john", "john smith");
        let ba = monge_elkan_default("john smith", "john");
        // "john" vs "john smith": john matches john perfectly → 1.0
        assert!(approx_eq(ab, 1.0));
        // "john smith" vs "john": john→john=1.0, smith→john<1.0 → avg < 1.0
        assert!(ba < 1.0);
    }

    #[test]
    fn custom_inner_metric() {
        use crate::metrics::Levenshtein;
        let me = MongeElkan {
            inner_metric: Box::new(Metric::Levenshtein(Levenshtein)),
        };
        let sim = me.similarity("john smith", "john smith");
        assert!(approx_eq(sim, 1.0));
    }
}
