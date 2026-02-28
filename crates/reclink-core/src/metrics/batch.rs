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

/// A row-major similarity matrix from an all-pairs comparison.
#[derive(Debug, Clone)]
pub struct ColumnarCdist {
    /// Flat row-major score buffer: `scores[i * n_cols + j]` = similarity(queries[i], candidates[j]).
    pub scores: Vec<f64>,
    /// Number of query strings (rows).
    pub n_rows: usize,
    /// Number of candidate strings (columns).
    pub n_cols: usize,
}

impl ColumnarCdist {
    /// Returns the similarity score for `(row, col)`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.scores[row * self.n_cols + col]
    }
}

/// Computes an all-pairs similarity matrix in a flat row-major layout.
///
/// Uses Rayon to parallelise over query rows.
#[must_use]
pub fn cdist_columnar(queries: &[&str], candidates: &[&str], metric: &Metric) -> ColumnarCdist {
    let n_rows = queries.len();
    let n_cols = candidates.len();

    if n_rows == 0 || n_cols == 0 {
        return ColumnarCdist {
            scores: Vec::new(),
            n_rows,
            n_cols,
        };
    }

    let scores: Vec<f64> = queries
        .par_iter()
        .flat_map(|q| {
            candidates
                .iter()
                .map(|c| metric.similarity(q, c))
                .collect::<Vec<f64>>()
        })
        .collect();

    ColumnarCdist {
        scores,
        n_rows,
        n_cols,
    }
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
    fn cdist_columnar_identity() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let strings = vec!["hello", "world"];
        let result = cdist_columnar(&strings, &strings, &metric);
        assert_eq!(result.n_rows, 2);
        assert_eq!(result.n_cols, 2);
        assert!((result.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((result.get(1, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cdist_columnar_dimensions() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let queries = vec!["a", "b", "c"];
        let candidates = vec!["x", "y"];
        let result = cdist_columnar(&queries, &candidates, &metric);
        assert_eq!(result.n_rows, 3);
        assert_eq!(result.n_cols, 2);
        assert_eq!(result.scores.len(), 6);
    }

    #[test]
    fn cdist_columnar_empty() {
        let metric = metric_from_name("jaro_winkler").unwrap();
        let empty: Vec<&str> = vec![];
        let result = cdist_columnar(&empty, &["hello"], &metric);
        assert_eq!(result.n_rows, 0);
        assert!(result.scores.is_empty());

        let result2 = cdist_columnar(&["hello"], &empty, &metric);
        assert_eq!(result2.n_cols, 0);
        assert!(result2.scores.is_empty());
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
