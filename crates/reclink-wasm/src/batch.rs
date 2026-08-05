//! Batch matching functions exposed to JavaScript.

use reclink_core::metrics::{batch, metric_from_name};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct JsMatchResult {
    index: usize,
    value: String,
    score: f64,
}

#[derive(Serialize)]
struct CdistResult {
    matrix: Vec<Vec<f64>>,
    rows: Vec<String>,
    cols: Vec<String>,
}

/// Find all candidates matching a query above a threshold.
///
/// `candidates` is a JS string array. Returns an array of `{index, value, score}`.
#[wasm_bindgen]
pub fn match_batch(
    query: &str,
    candidates: JsValue,
    metric: &str,
    threshold: f64,
) -> Result<JsValue, JsError> {
    let cands: Vec<String> =
        serde_wasm_bindgen::from_value(candidates).map_err(|e| JsError::new(&e.to_string()))?;
    let m = metric_from_name(metric).map_err(|e| JsError::new(&e.to_string()))?;
    let refs: Vec<&str> = cands.iter().map(String::as_str).collect();
    let thresh = if threshold <= 0.0 {
        None
    } else {
        Some(threshold)
    };

    let results = batch::match_batch(query, &refs, &m, thresh);
    let output: Vec<JsMatchResult> = results
        .into_iter()
        .map(|r| JsMatchResult {
            value: cands[r.index].clone(),
            index: r.index,
            score: r.score,
        })
        .collect();
    Ok(serde_wasm_bindgen::to_value(&output)?)
}

/// Find the single best match for a query.
///
/// `candidates` is a JS string array. Returns `{index, value, score}` or null.
#[wasm_bindgen]
pub fn match_best(
    query: &str,
    candidates: JsValue,
    metric: &str,
    threshold: f64,
) -> Result<JsValue, JsError> {
    let cands: Vec<String> =
        serde_wasm_bindgen::from_value(candidates).map_err(|e| JsError::new(&e.to_string()))?;
    let m = metric_from_name(metric).map_err(|e| JsError::new(&e.to_string()))?;
    let refs: Vec<&str> = cands.iter().map(String::as_str).collect();
    let thresh = if threshold <= 0.0 {
        None
    } else {
        Some(threshold)
    };

    match batch::match_best(query, &refs, &m, thresh) {
        Some(r) => {
            let output = JsMatchResult {
                value: cands[r.index].clone(),
                index: r.index,
                score: r.score,
            };
            Ok(serde_wasm_bindgen::to_value(&output)?)
        }
        None => Ok(JsValue::NULL),
    }
}

/// Compute an all-pairs similarity matrix.
///
/// `sources` and `targets` are JS string arrays. Returns `{matrix, rows, cols}`.
#[wasm_bindgen]
pub fn cdist(sources: JsValue, targets: JsValue, metric: &str) -> Result<JsValue, JsError> {
    let src: Vec<String> =
        serde_wasm_bindgen::from_value(sources).map_err(|e| JsError::new(&e.to_string()))?;
    let tgt: Vec<String> =
        serde_wasm_bindgen::from_value(targets).map_err(|e| JsError::new(&e.to_string()))?;
    let m = metric_from_name(metric).map_err(|e| JsError::new(&e.to_string()))?;
    let src_refs: Vec<&str> = src.iter().map(String::as_str).collect();
    let tgt_refs: Vec<&str> = tgt.iter().map(String::as_str).collect();

    let result = batch::cdist_columnar(&src_refs, &tgt_refs, &m);

    // Convert flat row-major to 2D matrix
    let mut matrix = Vec::with_capacity(result.n_rows);
    for i in 0..result.n_rows {
        let start = i * result.n_cols;
        let end = start + result.n_cols;
        matrix.push(result.scores[start..end].to_vec());
    }

    let output = CdistResult {
        matrix,
        rows: src,
        cols: tgt,
    };
    Ok(serde_wasm_bindgen::to_value(&output)?)
}
