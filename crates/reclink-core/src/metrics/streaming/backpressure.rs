//! Bounded-channel streaming matcher with backpressure support.
//!
//! Uses `crossbeam-channel` bounded channels so that the producer thread
//! blocks when the consumer falls behind, providing true backpressure.

use crossbeam_channel::{bounded, Receiver};

use crate::metrics::Metric;

/// A streaming matcher that scores candidates in a background thread
/// with bounded-channel backpressure.
pub struct BoundedStreamingMatcher {
    query: String,
    metric: Metric,
    threshold: Option<f64>,
    buffer_capacity: usize,
}

impl BoundedStreamingMatcher {
    /// Create a new bounded streaming matcher.
    ///
    /// `buffer_capacity` controls the maximum number of results buffered
    /// before the producer blocks.
    #[must_use]
    pub fn new(
        query: String,
        metric: Metric,
        threshold: Option<f64>,
        buffer_capacity: usize,
    ) -> Self {
        Self {
            query,
            metric,
            threshold,
            buffer_capacity,
        }
    }

    /// Score candidates in a background thread, returning a receiver.
    ///
    /// The sender blocks when the buffer is full (backpressure).
    /// Each result is `(candidate, score, original_index)`.
    pub fn stream_match(&self, candidates: Vec<String>) -> Receiver<(String, f64, usize)> {
        let (tx, rx) = bounded(self.buffer_capacity);
        let query = self.query.clone();
        let metric = self.metric.clone();
        let threshold = self.threshold;

        std::thread::spawn(move || {
            for (i, candidate) in candidates.into_iter().enumerate() {
                let score = metric.similarity(&query, &candidate);
                if threshold.is_none_or(|t| score >= t) && tx.send((candidate, score, i)).is_err() {
                    break; // receiver dropped
                }
            }
        });

        rx
    }

    /// Score a chunk of candidates with a global index offset.
    ///
    /// Collects all results synchronously (useful for Python interop).
    pub fn score_bounded(
        &self,
        candidates: Vec<String>,
        offset: usize,
    ) -> Vec<(String, f64, usize)> {
        let rx = self.stream_match(candidates);
        rx.into_iter()
            .map(|(s, score, i)| (s, score, i + offset))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::JaroWinkler;

    #[test]
    fn bounded_streaming_basic() {
        let matcher = BoundedStreamingMatcher::new(
            "hello".to_string(),
            Metric::JaroWinkler(JaroWinkler::default()),
            None,
            16,
        );
        let results = matcher.score_bounded(
            vec!["hello".to_string(), "world".to_string(), "help".to_string()],
            0,
        );
        assert_eq!(results.len(), 3);
        // First result should be exact match
        assert!((results[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bounded_streaming_with_threshold() {
        let matcher = BoundedStreamingMatcher::new(
            "hello".to_string(),
            Metric::JaroWinkler(JaroWinkler::default()),
            Some(0.9),
            16,
        );
        let results = matcher.score_bounded(
            vec!["hello".to_string(), "xyz".to_string(), "help".to_string()],
            0,
        );
        // Only "hello" should be above 0.9
        assert!(results.iter().any(|(s, _, _)| s == "hello"));
        assert!(!results.iter().any(|(s, _, _)| s == "xyz"));
    }

    #[test]
    fn bounded_streaming_with_offset() {
        let matcher = BoundedStreamingMatcher::new(
            "hello".to_string(),
            Metric::JaroWinkler(JaroWinkler::default()),
            None,
            16,
        );
        let results = matcher.score_bounded(vec!["hello".to_string(), "world".to_string()], 100);
        assert_eq!(results[0].2, 100); // offset applied
        assert_eq!(results[1].2, 101);
    }

    #[test]
    fn bounded_streaming_backpressure() {
        // Buffer capacity of 1 — producer blocks after 1 buffered result
        let matcher = BoundedStreamingMatcher::new(
            "hello".to_string(),
            Metric::JaroWinkler(JaroWinkler::default()),
            None,
            1,
        );
        let rx = matcher.stream_match(vec![
            "hello".to_string(),
            "world".to_string(),
            "help".to_string(),
        ]);
        // Consume all results
        let results: Vec<_> = rx.into_iter().collect();
        assert_eq!(results.len(), 3);
    }
}
