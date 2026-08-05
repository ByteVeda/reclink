//! Index structures exposed to JavaScript as stateful WASM classes.

use reclink_core::index::{BkTree, MinHashIndex, NgramIndex, VpTree};
use reclink_core::metrics::metric_from_name;
use serde::Serialize;
use wasm_bindgen::prelude::*;

// ── BK-Tree ──────────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct WasmBkTree {
    inner: BkTree,
    strings: Vec<String>,
}

#[derive(Serialize)]
struct BkResult {
    index: usize,
    value: String,
    distance: usize,
}

#[wasm_bindgen]
impl WasmBkTree {
    /// Build a BK-tree from a list of strings.
    ///
    /// `metric` must be a distance metric: "levenshtein", "damerau_levenshtein", or "hamming".
    pub fn build(strings: JsValue, metric: &str) -> Result<WasmBkTree, JsError> {
        let owned: Vec<String> =
            serde_wasm_bindgen::from_value(strings).map_err(|e| JsError::new(&e.to_string()))?;
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let m = metric_from_name(metric).map_err(|e| JsError::new(&e.to_string()))?;
        let tree = BkTree::build(&refs, m).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmBkTree {
            inner: tree,
            strings: owned,
        })
    }

    /// Find all strings within `max_distance` of the query.
    pub fn find_within(&self, query: &str, max_distance: usize) -> JsValue {
        let results = self.inner.find_within(query, max_distance);
        let output: Vec<BkResult> = results
            .into_iter()
            .map(|r| BkResult {
                value: self.strings[r.index].clone(),
                index: r.index,
                distance: r.distance,
            })
            .collect();
        serde_wasm_bindgen::to_value(&output).unwrap()
    }

    /// Find the k nearest neighbors of the query.
    pub fn find_nearest(&self, query: &str, k: usize) -> JsValue {
        let results = self.inner.find_nearest(query, k);
        let output: Vec<BkResult> = results
            .into_iter()
            .map(|r| BkResult {
                value: self.strings[r.index].clone(),
                index: r.index,
                distance: r.distance,
            })
            .collect();
        serde_wasm_bindgen::to_value(&output).unwrap()
    }

    /// Number of strings in the tree.
    pub fn len(&self) -> usize {
        self.strings.len()
    }
}

// ── VP-Tree ──────────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct WasmVpTree {
    inner: VpTree,
    strings: Vec<String>,
}

#[derive(Serialize)]
struct VpResult {
    index: usize,
    value: String,
    distance: f64,
}

#[wasm_bindgen]
impl WasmVpTree {
    /// Build a VP-tree from a list of strings using any similarity metric.
    pub fn build(strings: JsValue, metric: &str) -> Result<WasmVpTree, JsError> {
        let owned: Vec<String> =
            serde_wasm_bindgen::from_value(strings).map_err(|e| JsError::new(&e.to_string()))?;
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let m = metric_from_name(metric).map_err(|e| JsError::new(&e.to_string()))?;
        let tree = VpTree::build(&refs, m);
        Ok(WasmVpTree {
            inner: tree,
            strings: owned,
        })
    }

    /// Find all strings within `max_distance` (dissimilarity) of the query.
    pub fn find_within(&self, query: &str, max_distance: f64) -> JsValue {
        let results = self.inner.find_within(query, max_distance);
        let output: Vec<VpResult> = results
            .into_iter()
            .map(|r| VpResult {
                value: self.strings[r.index].clone(),
                index: r.index,
                distance: r.distance,
            })
            .collect();
        serde_wasm_bindgen::to_value(&output).unwrap()
    }

    /// Find the k nearest neighbors of the query.
    pub fn find_nearest(&self, query: &str, k: usize) -> JsValue {
        let results = self.inner.find_nearest(query, k);
        let output: Vec<VpResult> = results
            .into_iter()
            .map(|r| VpResult {
                value: self.strings[r.index].clone(),
                index: r.index,
                distance: r.distance,
            })
            .collect();
        serde_wasm_bindgen::to_value(&output).unwrap()
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }
}

// ── N-gram Index ─────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct WasmNgramIndex {
    inner: NgramIndex,
}

#[derive(Serialize)]
struct NgramResult {
    index: usize,
    value: String,
    shared: usize,
}

#[wasm_bindgen]
impl WasmNgramIndex {
    /// Build an n-gram index from a list of strings.
    pub fn build(strings: JsValue, n: usize) -> Result<WasmNgramIndex, JsError> {
        let owned: Vec<String> =
            serde_wasm_bindgen::from_value(strings).map_err(|e| JsError::new(&e.to_string()))?;
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let index = NgramIndex::build(&refs, n);
        Ok(WasmNgramIndex { inner: index })
    }

    /// Search for strings sharing at least `threshold` n-grams with the query.
    pub fn search(&self, query: &str, threshold: usize) -> JsValue {
        let results = self.inner.search(query, threshold);
        let output: Vec<NgramResult> = results
            .into_iter()
            .map(|r| NgramResult {
                value: r.value.clone(),
                index: r.index,
                shared: r.shared_ngrams,
            })
            .collect();
        serde_wasm_bindgen::to_value(&output).unwrap()
    }

    /// Find the top-k most similar strings by shared n-gram count.
    pub fn search_top_k(&self, query: &str, k: usize) -> JsValue {
        let results = self.inner.search_top_k(query, k);
        let output: Vec<NgramResult> = results
            .into_iter()
            .map(|r| NgramResult {
                value: r.value.clone(),
                index: r.index,
                shared: r.shared_ngrams,
            })
            .collect();
        serde_wasm_bindgen::to_value(&output).unwrap()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

// ── MinHash Index ────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct WasmMinHashIndex {
    inner: MinHashIndex,
}

#[derive(Serialize)]
struct MinHashResult {
    index: usize,
    value: String,
    similarity: f64,
}

#[wasm_bindgen]
impl WasmMinHashIndex {
    /// Build a MinHash LSH index from a list of strings.
    pub fn build(
        strings: JsValue,
        num_hashes: usize,
        num_bands: usize,
    ) -> Result<WasmMinHashIndex, JsError> {
        let owned: Vec<String> =
            serde_wasm_bindgen::from_value(strings).map_err(|e| JsError::new(&e.to_string()))?;
        let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let index = MinHashIndex::build(&refs, num_hashes, num_bands);
        Ok(WasmMinHashIndex { inner: index })
    }

    /// Query for similar strings above a Jaccard similarity threshold.
    pub fn query(&self, s: &str, threshold: f64) -> JsValue {
        let results = self.inner.query(s, threshold);
        let output: Vec<MinHashResult> = results
            .into_iter()
            .map(|(index, value, similarity)| MinHashResult {
                index,
                value,
                similarity,
            })
            .collect();
        serde_wasm_bindgen::to_value(&output).unwrap()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
