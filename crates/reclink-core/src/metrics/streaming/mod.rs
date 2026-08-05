//! Streaming matcher for lazy/chunked candidate processing.

#[cfg(feature = "streaming-backpressure")]
pub mod backpressure;

use crate::metrics::Metric;

/// Streaming matcher: scores candidates one at a time or in chunks.
///
/// Useful for processing large candidate sets without materializing
/// everything in memory at once.
pub struct StreamingMatcher {
    query: String,
    metric: Metric,
    threshold: Option<f64>,
}

impl StreamingMatcher {
    /// Create a new streaming matcher.
    #[must_use]
    pub fn new(query: String, metric: Metric, threshold: Option<f64>) -> Self {
        Self {
            query,
            metric,
            threshold,
        }
    }

    /// Score a single candidate. Returns `Some(score)` if above threshold.
    #[must_use]
    pub fn score(&self, candidate: &str) -> Option<f64> {
        let s = self.metric.similarity(&self.query, candidate);
        match self.threshold {
            Some(t) if s < t => None,
            _ => Some(s),
        }
    }

    /// Score a batch of candidates. Returns `(index_in_chunk, score)` pairs
    /// for candidates above threshold.
    #[must_use]
    pub fn score_batch(&self, candidates: &[&str]) -> Vec<(usize, f64)> {
        candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| self.score(c).map(|s| (i, s)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{metric_from_name, JaroWinkler};

    #[test]
    fn score_no_threshold() {
        let matcher = StreamingMatcher::new(
            "hello".to_string(),
            Metric::JaroWinkler(JaroWinkler::default()),
            None,
        );
        let score = matcher.score("hello");
        assert!(score.is_some());
        assert!((score.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn score_with_threshold() {
        let matcher = StreamingMatcher::new(
            "hello".to_string(),
            Metric::JaroWinkler(JaroWinkler::default()),
            Some(0.9),
        );
        // "hello" vs "hello" = 1.0, above threshold
        assert!(matcher.score("hello").is_some());
        // "hello" vs "xyz" = low, below threshold
        assert!(matcher.score("xyz").is_none());
    }

    #[test]
    fn score_batch_results() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let matcher = StreamingMatcher::new("hello".to_string(), metric, Some(0.5));
        let results = matcher.score_batch(&["hello", "xyz", "help", "abc"]);
        // "hello" and "help" should be above 0.5
        assert!(results.len() >= 2);
        assert!(results.iter().any(|(i, _)| *i == 0)); // "hello"
        assert!(results.iter().any(|(i, _)| *i == 2)); // "help"
    }
}
