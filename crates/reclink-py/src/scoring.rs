use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reclink_core::metrics;

/// TF-IDF weighted string matcher.
///
/// Must be fit on a corpus before use. Down-weights common tokens like "the", "inc", etc.
#[pyclass]
struct PyTfIdfMatcher {
    inner: reclink_core::metrics::tfidf::TfIdfMatcher,
}

#[pymethods]
impl PyTfIdfMatcher {
    /// Fit the matcher on a corpus of strings.
    #[staticmethod]
    fn fit(corpus: Vec<String>) -> Self {
        let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        Self {
            inner: reclink_core::metrics::tfidf::TfIdfMatcher::fit(&refs),
        }
    }

    /// Compute TF-IDF cosine similarity between two strings.
    fn similarity(&self, a: &str, b: &str) -> f64 {
        self.inner.similarity(a, b)
    }

    /// Find all matches for a query among candidates.
    ///
    /// Returns list of (matched_string, score, index) tuples sorted by descending score.
    #[pyo3(signature = (query, candidates, threshold=None))]
    fn match_batch(
        &self,
        query: &str,
        candidates: Vec<String>,
        threshold: Option<f64>,
    ) -> Vec<(String, f64, usize)> {
        let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
        self.inner
            .match_batch(query, &refs, threshold)
            .into_iter()
            .map(|r| (candidates[r.index].clone(), r.score, r.index))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.n_docs()
    }
}

/// A weighted combination of multiple metrics.
///
/// Combines multiple string similarity metrics with configurable weights.
/// Weights are normalized to sum to 1.0 automatically.
#[pyclass]
struct PyCompositeScorer {
    inner: reclink_core::metrics::CompositeScorer,
}

#[pymethods]
impl PyCompositeScorer {
    /// Create a new CompositeScorer from a list of (metric_name, weight) pairs.
    #[new]
    fn new(components: Vec<(String, f64)>) -> PyResult<Self> {
        let pairs: Vec<(&str, f64)> = components.iter().map(|(n, w)| (n.as_str(), *w)).collect();
        let scorer = reclink_core::metrics::CompositeScorer::from_names(&pairs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: scorer })
    }

    /// Create a preset scorer by name.
    ///
    /// Available presets: "name_matching", "address_matching", "general_purpose".
    #[staticmethod]
    fn preset(name: &str) -> PyResult<Self> {
        let scorer = reclink_core::metrics::CompositeScorer::preset(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: scorer })
    }

    /// Compute the weighted composite similarity score.
    fn similarity(&self, a: &str, b: &str) -> f64 {
        self.inner.similarity(a, b)
    }

    /// Find the best match for a query among candidates.
    #[pyo3(signature = (query, candidates, threshold=None))]
    fn match_best(
        &self,
        query: &str,
        candidates: Vec<String>,
        threshold: Option<f64>,
    ) -> Option<(String, f64, usize)> {
        let mut best: Option<(String, f64, usize)> = None;
        for (i, c) in candidates.iter().enumerate() {
            let score = self.inner.similarity(query, c);
            if let Some(t) = threshold {
                if score < t {
                    continue;
                }
            }
            if best.as_ref().is_none_or(|(_, s, _)| score > *s) {
                best = Some((c.clone(), score, i));
            }
        }
        best
    }

    /// Find all matches above threshold, sorted by descending score.
    #[pyo3(signature = (query, candidates, threshold=None, limit=None))]
    fn match_batch(
        &self,
        query: &str,
        candidates: Vec<String>,
        threshold: Option<f64>,
        limit: Option<usize>,
    ) -> Vec<(String, f64, usize)> {
        let mut results: Vec<(String, f64, usize)> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let score = self.inner.similarity(query, c);
                if let Some(t) = threshold {
                    if score < t {
                        return None;
                    }
                }
                Some((c.clone(), score, i))
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if let Some(lim) = limit {
            results.truncate(lim);
        }
        results
    }
}

/// Streaming matcher for scoring candidates one at a time or in chunks.
///
/// Useful for processing large candidate sets lazily from iterators.
#[pyclass]
struct PyStreamingMatcher {
    inner: reclink_core::metrics::StreamingMatcher,
}

#[pymethods]
impl PyStreamingMatcher {
    /// Create a new streaming matcher.
    #[new]
    #[pyo3(signature = (query, scorer="jaro_winkler", threshold=None))]
    fn new(query: String, scorer: &str, threshold: Option<f64>) -> PyResult<Self> {
        let metric =
            metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: reclink_core::metrics::StreamingMatcher::new(query, metric, threshold),
        })
    }

    /// Score a single candidate. Returns None if below threshold.
    fn score(&self, candidate: &str) -> Option<f64> {
        self.inner.score(candidate)
    }

    /// Score a chunk of candidates. Returns list of (index_in_chunk, score).
    fn score_chunk(&self, candidates: Vec<String>) -> Vec<(usize, f64)> {
        let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
        self.inner.score_batch(&refs)
    }
}

/// Bounded-channel streaming matcher with backpressure.
///
/// Scores candidates in a background thread with a bounded buffer.
/// When the buffer is full, the producer thread blocks until the consumer
/// reads results, providing true backpressure.
#[pyclass]
struct PyBoundedStreamingMatcher {
    inner: reclink_core::metrics::streaming::backpressure::BoundedStreamingMatcher,
}

#[pymethods]
impl PyBoundedStreamingMatcher {
    /// Create a new bounded streaming matcher.
    #[new]
    #[pyo3(signature = (query, scorer="jaro_winkler", threshold=None, buffer_size=64))]
    fn new(
        query: String,
        scorer: &str,
        threshold: Option<f64>,
        buffer_size: usize,
    ) -> PyResult<Self> {
        let metric =
            metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: reclink_core::metrics::streaming::backpressure::BoundedStreamingMatcher::new(
                query,
                metric,
                threshold,
                buffer_size,
            ),
        })
    }

    /// Score a chunk of candidates with a global index offset.
    ///
    /// Returns list of (matched_string, score, global_index) tuples.
    #[pyo3(signature = (candidates, offset=0))]
    fn score_bounded(
        &self,
        py: Python<'_>,
        candidates: Vec<String>,
        offset: usize,
    ) -> Vec<(String, f64, usize)> {
        py.allow_threads(|| self.inner.score_bounded(candidates, offset))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTfIdfMatcher>()?;
    m.add_class::<PyCompositeScorer>()?;
    m.add_class::<PyStreamingMatcher>()?;
    m.add_class::<PyBoundedStreamingMatcher>()?;
    Ok(())
}
