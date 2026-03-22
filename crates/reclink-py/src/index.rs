use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reclink_core::index;
use reclink_core::metrics;

fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

/// BK-tree for efficient metric-space nearest-neighbor search.
///
/// Only works with integer distance metrics (levenshtein, damerau_levenshtein, hamming).
#[pyclass]
struct PyBkTree {
    inner: reclink_core::index::BkTree,
}

#[pymethods]
impl PyBkTree {
    /// Build a new BK-tree from a list of strings.
    #[staticmethod]
    fn build(strings: Vec<String>, metric: &str) -> PyResult<Self> {
        let m =
            metrics::metric_from_name(metric).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        let tree = reclink_core::index::BkTree::build(&refs, m)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: tree })
    }

    /// Find all strings within max_distance of the query.
    ///
    /// Returns list of (value, index, distance) tuples.
    fn find_within(&self, query: &str, max_distance: usize) -> Vec<(String, usize, usize)> {
        self.inner
            .find_within(query, max_distance)
            .into_iter()
            .map(|r| (r.value, r.index, r.distance))
            .collect()
    }

    /// Find the k nearest neighbors.
    ///
    /// Returns list of (value, index, distance) tuples.
    fn find_nearest(&self, query: &str, k: usize) -> Vec<(String, usize, usize)> {
        self.inner
            .find_nearest(query, k)
            .into_iter()
            .map(|r| (r.value, r.index, r.distance))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Insert a new string and return its assigned index.
    fn insert(&mut self, s: &str) -> usize {
        self.inner.insert_new(s)
    }

    /// Soft-delete a string by index.
    fn remove(&mut self, index: usize) -> bool {
        self.inner.remove(index)
    }

    /// Check if an index is valid and not deleted.
    fn __contains__(&self, index: usize) -> bool {
        self.inner.contains(index)
    }

    /// Estimated heap memory usage in bytes.
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Estimated heap memory usage as a human-readable string.
    fn memory_usage_human(&self) -> String {
        format_bytes(self.inner.memory_usage())
    }

    /// Save the BK-tree to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        index::persistence::save_to_file(&self.inner, std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load a BK-tree from a file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner: reclink_core::index::BkTree =
            index::persistence::load_from_file(std::path::Path::new(path))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

/// Vantage-point tree for efficient nearest-neighbor search.
///
/// Works with any similarity metric (unlike BK-tree which requires integer distances).
#[pyclass]
struct PyVpTree {
    inner: index::VpTree,
}

#[pymethods]
impl PyVpTree {
    /// Build a VP-tree from a list of strings.
    #[staticmethod]
    fn build(strings: Vec<String>, metric: &str) -> PyResult<Self> {
        let m =
            metrics::metric_from_name(metric).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        let tree = index::VpTree::build(&refs, m);
        Ok(Self { inner: tree })
    }

    /// Find all strings within max_distance (dissimilarity) of the query.
    fn find_within(&self, query: &str, max_distance: f64) -> Vec<(String, usize, f64)> {
        self.inner
            .find_within(query, max_distance)
            .into_iter()
            .map(|r| (r.value, r.index, r.distance))
            .collect()
    }

    /// Find the k nearest neighbors of the query.
    fn find_nearest(&self, query: &str, k: usize) -> Vec<(String, usize, f64)> {
        self.inner
            .find_nearest(query, k)
            .into_iter()
            .map(|r| (r.value, r.index, r.distance))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Insert a new string into the buffer and return its assigned index.
    fn insert(&mut self, s: &str) -> usize {
        self.inner.insert_new(s)
    }

    /// Soft-delete a string by index.
    fn remove(&mut self, index: usize) -> bool {
        self.inner.remove(index)
    }

    /// Check if an index is valid and not deleted.
    fn __contains__(&self, index: usize) -> bool {
        self.inner.contains(index)
    }

    /// Rebuild the tree, consolidating buffer items and removing deleted entries.
    fn rebuild(&mut self) {
        self.inner.rebuild();
    }

    /// Estimated heap memory usage in bytes.
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Estimated heap memory usage as a human-readable string.
    fn memory_usage_human(&self) -> String {
        format_bytes(self.inner.memory_usage())
    }

    /// Save the VP-tree to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        index::persistence::save_to_file(&self.inner, std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load a VP-tree from a file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner: index::VpTree = index::persistence::load_from_file(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

/// Inverted n-gram index for fast approximate string matching.
///
/// Maps character n-grams to candidate strings. Useful for quickly finding
/// strings that share many n-grams with a query.
#[pyclass]
struct PyNgramIndex {
    inner: index::NgramIndex,
}

#[pymethods]
impl PyNgramIndex {
    /// Build an n-gram index from a list of strings.
    #[staticmethod]
    fn build(strings: Vec<String>, n: usize) -> Self {
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        Self {
            inner: index::NgramIndex::build(&refs, n),
        }
    }

    /// Find all strings sharing at least `threshold` n-grams with the query.
    fn search(&self, query: &str, threshold: usize) -> Vec<(String, usize, usize)> {
        self.inner
            .search(query, threshold)
            .into_iter()
            .map(|r| (r.value, r.index, r.shared_ngrams))
            .collect()
    }

    /// Find the k strings sharing the most n-grams with the query.
    fn search_top_k(&self, query: &str, k: usize) -> Vec<(String, usize, usize)> {
        self.inner
            .search_top_k(query, k)
            .into_iter()
            .map(|r| (r.value, r.index, r.shared_ngrams))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Insert a new string and return its assigned index.
    fn insert(&mut self, s: &str) -> usize {
        self.inner.insert_new(s)
    }

    /// Soft-delete a string by index.
    fn remove(&mut self, index: usize) -> bool {
        self.inner.remove(index)
    }

    /// Check if an index is valid and not deleted.
    fn __contains__(&self, index: usize) -> bool {
        self.inner.contains(index)
    }

    /// Estimated heap memory usage in bytes.
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Estimated heap memory usage as a human-readable string.
    fn memory_usage_human(&self) -> String {
        format_bytes(self.inner.memory_usage())
    }

    /// Save the n-gram index to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        index::persistence::save_to_file(&self.inner, std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load an n-gram index from a file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner: index::NgramIndex =
            index::persistence::load_from_file(std::path::Path::new(path))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

/// Memory-mapped N-gram index for datasets larger than RAM.
///
/// Build an index once with `build_and_save()`, then open it with `open()`.
/// Queries operate directly on memory-mapped data without loading the full
/// index into the heap.
#[pyclass]
struct PyMmapNgramIndex {
    inner: index::MmapNgramIndex,
}

#[pymethods]
impl PyMmapNgramIndex {
    /// Build an index from strings and save to a file.
    #[staticmethod]
    fn build_and_save(strings: Vec<String>, n: usize, path: &str) -> PyResult<()> {
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        index::MmapNgramIndex::build_and_save(&refs, n, std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Open a memory-mapped index from a file.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let inner = index::MmapNgramIndex::open(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Find all strings sharing at least `threshold` n-grams with the query.
    fn search(&self, query: &str, threshold: usize) -> Vec<(String, usize, usize)> {
        self.inner
            .search(query, threshold)
            .into_iter()
            .map(|r| (r.value, r.index, r.shared_ngrams))
            .collect()
    }

    /// Find the k strings sharing the most n-grams with the query.
    fn search_top_k(&self, query: &str, k: usize) -> Vec<(String, usize, usize)> {
        self.inner
            .search_top_k(query, k)
            .into_iter()
            .map(|r| (r.value, r.index, r.shared_ngrams))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Estimated heap memory usage in bytes.
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Estimated heap memory usage as a human-readable string.
    fn memory_usage_human(&self) -> String {
        format_bytes(self.inner.memory_usage())
    }
}

/// MinHash/LSH index for approximate nearest-neighbor search.
///
/// Uses MinHash signatures and banding for efficient similarity search.
#[pyclass]
struct PyMinHashIndex {
    inner: index::MinHashIndex,
}

#[pymethods]
impl PyMinHashIndex {
    /// Build a new MinHash index from a list of strings.
    #[staticmethod]
    #[pyo3(signature = (strings, num_hashes=100, num_bands=20, shingle_size=3))]
    fn build(
        strings: Vec<String>,
        num_hashes: usize,
        num_bands: usize,
        shingle_size: usize,
    ) -> Self {
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        Self {
            inner: index::MinHashIndex::build_with_shingle_size(
                &refs,
                num_hashes,
                num_bands,
                shingle_size,
            ),
        }
    }

    /// Find all similar strings above the given threshold.
    ///
    /// Returns list of (index, value, estimated_similarity) tuples,
    /// sorted by descending similarity.
    #[pyo3(signature = (query, threshold=0.5))]
    fn query(&self, query: &str, threshold: f64) -> Vec<(usize, String, f64)> {
        self.inner.query(query, threshold)
    }

    /// Insert a new string and return its assigned index.
    fn insert(&mut self, s: &str) -> usize {
        self.inner.insert(s)
    }

    /// Soft-delete a string by index.
    fn remove(&mut self, index: usize) -> bool {
        self.inner.remove(index)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Estimated heap memory usage in bytes.
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// Estimated heap memory usage as a human-readable string.
    fn memory_usage_human(&self) -> String {
        format_bytes(self.inner.memory_usage())
    }

    /// Save the index to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        index::persistence::save_to_file(&self.inner, std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load an index from a file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let mut inner: index::MinHashIndex =
            index::persistence::load_from_file(std::path::Path::new(path))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        inner.rebuild_buckets();
        Ok(Self { inner })
    }
}

/// Bloom filter for fast probabilistic membership testing.
#[pyclass]
struct PyBloomFilter {
    inner: reclink_core::index::BloomFilter,
}

#[pymethods]
impl PyBloomFilter {
    /// Create a new Bloom filter.
    #[new]
    #[pyo3(signature = (expected_items=1000, false_positive_rate=0.01))]
    fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        Self {
            inner: reclink_core::index::BloomFilter::with_capacity(
                expected_items,
                false_positive_rate,
            ),
        }
    }

    /// Insert a string into the filter.
    fn insert(&mut self, item: &str) {
        self.inner.insert(item);
    }

    /// Test if a string may be in the filter (no false negatives).
    fn contains(&self, item: &str) -> bool {
        self.inner.contains(item)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, item: &str) -> bool {
        self.inner.contains(item)
    }

    /// Approximate memory usage in bytes.
    fn memory_usage(&self) -> String {
        format_bytes(self.inner.memory_usage())
    }

    /// Estimated false positive rate given current count.
    fn estimated_fp_rate(&self) -> f64 {
        self.inner.estimated_fp_rate()
    }
}

/// Inverted index for token-based candidate retrieval.
#[pyclass]
struct PyInvertedIndex {
    inner: reclink_core::index::InvertedIndex,
}

#[pymethods]
impl PyInvertedIndex {
    /// Build an inverted index from strings.
    ///
    /// Parameters
    /// ----------
    /// strings : list of str
    ///     Strings to index.
    /// tokenizer : str
    ///     Tokenization method: "whitespace" or "ngram:N" (e.g. "ngram:2").
    #[staticmethod]
    #[pyo3(signature = (strings, tokenizer="whitespace"))]
    fn build(strings: Vec<String>, tokenizer: &str) -> PyResult<Self> {
        let tok_kind = parse_tokenizer(tokenizer)?;
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        Ok(Self {
            inner: reclink_core::index::InvertedIndex::build(&refs, tok_kind),
        })
    }

    /// Search for records sharing at least min_shared tokens with query.
    fn search(&self, query: &str, min_shared: usize) -> Vec<(String, usize, usize)> {
        self.inner
            .search(query, min_shared)
            .into_iter()
            .map(|r| (r.value, r.index, r.shared_tokens))
            .collect()
    }

    /// Return top-k records by shared token count.
    fn search_top_k(&self, query: &str, k: usize) -> Vec<(String, usize, usize)> {
        self.inner
            .search_top_k(query, k)
            .into_iter()
            .map(|r| (r.value, r.index, r.shared_tokens))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of unique tokens in the index.
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

fn parse_tokenizer(s: &str) -> PyResult<reclink_core::index::TokenizerKind> {
    if s == "whitespace" {
        Ok(reclink_core::index::TokenizerKind::Whitespace)
    } else if let Some(n_str) = s.strip_prefix("ngram:") {
        let n: usize = n_str
            .parse()
            .map_err(|_| PyValueError::new_err(format!("invalid ngram size: {n_str}")))?;
        Ok(reclink_core::index::TokenizerKind::Ngram(n))
    } else {
        Err(PyValueError::new_err(format!(
            "unknown tokenizer: {s}. Expected: 'whitespace' or 'ngram:N'"
        )))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBkTree>()?;
    m.add_class::<PyVpTree>()?;
    m.add_class::<PyNgramIndex>()?;
    m.add_class::<PyMmapNgramIndex>()?;
    m.add_class::<PyMinHashIndex>()?;
    m.add_class::<PyBloomFilter>()?;
    m.add_class::<PyInvertedIndex>()?;
    Ok(())
}
