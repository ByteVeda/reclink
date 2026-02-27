//! Arrow-friendly batch operations for reclink.
//!
//! Provides batch similarity computation and matching functions designed
//! for use with Arrow-based DataFrame libraries (PyArrow, Polars, pandas).
//! These functions accept string arrays and return results in array-friendly
//! formats, minimizing Python ↔ Rust round-trips.
//!
//! Requires the `arrow-interop` feature flag.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use reclink_core::metrics;
use reclink_core::phonetic::{self as phonetic_mod, PhoneticEncoder};

/// Compute an all-pairs similarity matrix from two string arrays.
///
/// Returns a flat row-major list of ``len(a) * len(b)`` similarity scores.
/// This is more efficient than `cdist` for Arrow workflows since it avoids
/// constructing a numpy array.
#[pyfunction]
#[pyo3(signature = (a, b, scorer="jaro_winkler"))]
pub fn cdist_arrow(
    py: Python<'_>,
    a: Vec<String>,
    b: Vec<String>,
    scorer: &str,
) -> PyResult<Vec<f64>> {
    let metric =
        metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let metric_ref = &metric;
    let result: Vec<f64> = py.allow_threads(|| {
        a.par_iter()
            .flat_map_iter(|s_a| b.iter().map(move |s_b| metric_ref.similarity(s_a, s_b)))
            .collect()
    });

    Ok(result)
}

/// Find the best match for a query in an array of candidates.
///
/// Returns ``(matched_string, score, index)`` or None if no match
/// exceeds the threshold.
#[pyfunction]
#[pyo3(signature = (query, candidates, scorer="jaro_winkler", threshold=0.0))]
pub fn match_best_arrow(
    query: &str,
    candidates: Vec<String>,
    scorer: &str,
    threshold: f64,
) -> PyResult<Option<(String, f64, usize)>> {
    let metric =
        metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let best = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (c, metric.similarity(query, c), i))
        .filter(|(_, score, _)| *score >= threshold)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(best.map(|(s, score, idx)| (s.clone(), score, idx)))
}

/// Find all matches above a threshold for a query in an array of candidates.
///
/// Returns a list of ``(matched_string, score, index)`` tuples, sorted by
/// score descending.
#[pyfunction]
#[pyo3(signature = (query, candidates, scorer="jaro_winkler", threshold=0.0, limit=None))]
pub fn match_batch_arrow(
    query: &str,
    candidates: Vec<String>,
    scorer: &str,
    threshold: f64,
    limit: Option<usize>,
) -> PyResult<Vec<(String, f64, usize)>> {
    let metric =
        metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut results: Vec<(String, f64, usize)> = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let score = metric.similarity(query, c);
            if score >= threshold {
                Some((c.clone(), score, i))
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(lim) = limit {
        results.truncate(lim);
    }

    Ok(results)
}

/// Compute element-wise similarity between two aligned string arrays.
///
/// Both arrays must have the same length. Returns a list of similarity scores.
#[pyfunction]
#[pyo3(signature = (a, b, scorer="jaro_winkler"))]
pub fn pairwise_similarity(
    py: Python<'_>,
    a: Vec<String>,
    b: Vec<String>,
    scorer: &str,
) -> PyResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "arrays must have equal length: {} != {}",
            a.len(),
            b.len()
        )));
    }

    let metric =
        metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result: Vec<f64> = py.allow_threads(|| {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(s_a, s_b)| metric.similarity(s_a, s_b))
            .collect()
    });

    Ok(result)
}

/// Batch phonetic encoding of a string array.
///
/// Returns a list of phonetic codes.
#[pyfunction]
#[pyo3(signature = (strings, algorithm="soundex"))]
pub fn phonetic_batch_arrow(
    py: Python<'_>,
    strings: Vec<String>,
    algorithm: &str,
) -> PyResult<Vec<String>> {
    let encoder: Box<dyn PhoneticEncoder + Send + Sync> = match algorithm {
        "soundex" => Box::new(phonetic_mod::Soundex),
        "metaphone" => Box::new(phonetic_mod::Metaphone),
        "nysiis" => Box::new(phonetic_mod::Nysiis),
        "caverphone" => Box::new(phonetic_mod::Caverphone),
        "cologne_phonetic" => Box::new(phonetic_mod::ColognePhonetic),
        "beider_morse" => Box::new(phonetic_mod::BeiderMorse::new()),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown phonetic algorithm: {other}"
            )));
        }
    };

    let result: Vec<String> = py.allow_threads(|| {
        strings.par_iter().map(|s| encoder.encode(s)).collect()
    });

    Ok(result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cdist_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(match_best_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(match_batch_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(phonetic_batch_arrow, m)?)?;
    Ok(())
}
