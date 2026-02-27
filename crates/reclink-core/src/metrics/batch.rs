//! Batch matching: one query vs many candidates.
//!
//! Provides `match_batch` (all results sorted by score) and `match_best` (single best match).

use rayon::prelude::*;

use crate::metrics::Metric;

/// A single match result from a batch comparison.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Index of the candidate in the original slice.
    pub index: usize,
    /// Similarity score in [0, 1].
    pub score: f64,
}

/// Returns all candidates scored against `query`, sorted by descending score.
///
/// If `threshold` is provided, only candidates with score >= threshold are returned.
#[must_use]
pub fn match_batch(
    query: &str,
    candidates: &[&str],
    metric: &Metric,
    threshold: Option<f64>,
) -> Vec<MatchResult> {
    let mut results: Vec<MatchResult> = candidates
        .par_iter()
        .enumerate()
        .map(|(i, c)| MatchResult {
            index: i,
            score: metric.similarity(query, c),
        })
        .filter(|r| threshold.is_none_or(|t| r.score >= t))
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Returns the single best match (highest score), or `None` if candidates is empty
/// or no candidate meets the threshold.
#[must_use]
pub fn match_best(
    query: &str,
    candidates: &[&str],
    metric: &Metric,
    threshold: Option<f64>,
) -> Option<MatchResult> {
    let best = candidates
        .par_iter()
        .enumerate()
        .map(|(i, c)| MatchResult {
            index: i,
            score: metric.similarity(query, c),
        })
        .reduce_with(|a, b| if a.score >= b.score { a } else { b });

    best.filter(|r| threshold.is_none_or(|t| r.score >= t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::metric_from_name;

    #[test]
    fn match_best_identical() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let candidates = vec!["hello", "world", "help"];
        let result = match_best("hello", &candidates, &metric, None).unwrap();
        assert_eq!(result.index, 0);
        assert!((result.score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn match_best_threshold_filters() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let candidates = vec!["xyz", "abc"];
        let result = match_best("hello", &candidates, &metric, Some(0.9));
        assert!(result.is_none());
    }

    #[test]
    fn match_best_empty() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let candidates: Vec<&str> = vec![];
        let result = match_best("hello", &candidates, &metric, None);
        assert!(result.is_none());
    }

    #[test]
    fn match_batch_sorted() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let candidates = vec!["world", "hello", "help"];
        let results = match_batch("hello", &candidates, &metric, None);
        assert_eq!(results.len(), 3);
        // First result should be the exact match
        assert_eq!(results[0].index, 1);
        // Results should be in descending score order
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn match_batch_threshold() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let candidates = vec!["hello", "world", "xyz"];
        let results = match_batch("hello", &candidates, &metric, Some(0.5));
        // "xyz" should be filtered out
        assert!(results.iter().all(|r| r.score >= 0.5));
    }

    #[test]
    fn match_batch_multiple_scorers() {
        for scorer in &["levenshtein", "jaro", "cosine", "jaccard"] {
            let metric = metric_from_name(scorer).unwrap();
            let candidates = vec!["hello", "world"];
            let results = match_batch("hello", &candidates, &metric, None);
            assert_eq!(results.len(), 2);
        }
    }
}
