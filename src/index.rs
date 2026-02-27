use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reclink_core::index;
use reclink_core::metrics;

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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBkTree>()?;
    m.add_class::<PyVpTree>()?;
    m.add_class::<PyNgramIndex>()?;
    m.add_class::<PyMmapNgramIndex>()?;
    m.add_class::<PyMinHashIndex>()?;
    Ok(())
}
