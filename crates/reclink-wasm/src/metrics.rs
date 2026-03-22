//! String metric functions exposed to JavaScript.

use reclink_core::metrics::{explain, metric_from_name, DistanceMetric};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Compute similarity between two strings using the named metric.
///
/// Returns a score in [0, 1] where 1 means identical.
#[wasm_bindgen]
pub fn similarity(a: &str, b: &str, metric: &str) -> Result<f64, JsError> {
    let m = metric_from_name(metric).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(m.similarity(a, b))
}

/// Compute raw edit distance between two strings (levenshtein, damerau_levenshtein, hamming).
#[wasm_bindgen]
pub fn distance(a: &str, b: &str, metric: &str) -> Result<JsValue, JsError> {
    let m = metric_from_name(metric).map_err(|e| JsError::new(&e.to_string()))?;
    match &m {
        reclink_core::metrics::Metric::Levenshtein(d) => {
            let dist = d.distance(a, b).map_err(|e| JsError::new(&e.to_string()))?;
            Ok(serde_wasm_bindgen::to_value(&dist)?)
        }
        reclink_core::metrics::Metric::DamerauLevenshtein(d) => {
            let dist = d.distance(a, b).map_err(|e| JsError::new(&e.to_string()))?;
            Ok(serde_wasm_bindgen::to_value(&dist)?)
        }
        reclink_core::metrics::Metric::Hamming(d) => {
            let dist = d.distance(a, b).map_err(|e| JsError::new(&e.to_string()))?;
            Ok(serde_wasm_bindgen::to_value(&dist)?)
        }
        _ => Err(JsError::new(&format!(
            "'{metric}' is a similarity metric, not a distance metric"
        ))),
    }
}

#[derive(Serialize)]
struct ExplainScore {
    algorithm: String,
    score: f64,
}

#[derive(Serialize)]
struct ExplainOutput {
    a: String,
    b: String,
    scores: Vec<ExplainScore>,
}

/// Compare two strings with all available metrics and return the breakdown.
#[wasm_bindgen]
pub fn explain_all(a: &str, b: &str) -> Result<JsValue, JsError> {
    let result = explain::explain(a, b);
    let output = ExplainOutput {
        a: result.a,
        b: result.b,
        scores: result
            .scores
            .into_iter()
            .map(|s| ExplainScore {
                algorithm: s.algorithm,
                score: s.score,
            })
            .collect(),
    };
    Ok(serde_wasm_bindgen::to_value(&output)?)
}

/// List all available metric names.
#[wasm_bindgen]
pub fn list_metrics() -> JsValue {
    let names = vec![
        "levenshtein",
        "damerau_levenshtein",
        "hamming",
        "jaro",
        "jaro_winkler",
        "cosine",
        "jaccard",
        "sorensen_dice",
        "weighted_levenshtein",
        "token_sort",
        "token_set",
        "partial_ratio",
        "lcs",
        "longest_common_substring",
        "ngram_similarity",
        "smith_waterman",
        "phonetic_hybrid",
        "ratcliff_obershelp",
        "needleman_wunsch",
        "gotoh",
        "monge_elkan",
    ];
    serde_wasm_bindgen::to_value(&names).unwrap()
}
