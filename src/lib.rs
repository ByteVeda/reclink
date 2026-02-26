//! PyO3 bindings for the reclink library.
//!
//! Exposes string similarity metrics, phonetic algorithms, preprocessing,
//! and a record linkage pipeline to Python.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use reclink_core::classify::Classifier;
use reclink_core::metrics::{
    self, Cosine, DamerauLevenshtein, DistanceMetric, Hamming, Jaccard, Jaro, JaroWinkler,
    Levenshtein, SimilarityMetric, SorensenDice,
};
use reclink_core::phonetic::{self as phonetic_mod, PhoneticEncoder};
use reclink_core::preprocess;

// ---------------------------------------------------------------------------
// String metrics
// ---------------------------------------------------------------------------

/// Compute Levenshtein edit distance between two strings.
#[pyfunction]
fn levenshtein(a: &str, b: &str) -> usize {
    reclink_core::metrics::levenshtein::levenshtein_distance(a, b)
}

/// Compute normalized Levenshtein similarity in [0, 1].
#[pyfunction]
fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    Levenshtein.normalized_similarity(a, b).unwrap_or(0.0)
}

/// Compute Damerau-Levenshtein distance between two strings.
#[pyfunction]
fn damerau_levenshtein(a: &str, b: &str) -> usize {
    reclink_core::metrics::damerau_levenshtein::damerau_levenshtein_distance(a, b)
}

/// Compute Damerau-Levenshtein normalized similarity in [0, 1].
#[pyfunction]
fn damerau_levenshtein_similarity(a: &str, b: &str) -> f64 {
    DamerauLevenshtein
        .normalized_similarity(a, b)
        .unwrap_or(0.0)
}

/// Compute Hamming distance between two equal-length strings.
#[pyfunction]
fn hamming(a: &str, b: &str) -> PyResult<usize> {
    reclink_core::metrics::hamming::hamming_distance(a, b)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute Hamming normalized similarity in [0, 1].
#[pyfunction]
fn hamming_similarity(a: &str, b: &str) -> PyResult<f64> {
    Hamming
        .normalized_similarity(a, b)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute Jaro similarity between two strings.
#[pyfunction]
fn jaro(a: &str, b: &str) -> f64 {
    Jaro.similarity(a, b)
}

/// Compute Jaro-Winkler similarity between two strings.
#[pyfunction]
#[pyo3(signature = (a, b, prefix_weight=0.1))]
fn jaro_winkler(a: &str, b: &str, prefix_weight: f64) -> f64 {
    JaroWinkler { prefix_weight }.similarity(a, b)
}

/// Compute cosine similarity between character n-gram vectors.
#[pyfunction]
#[pyo3(signature = (a, b, n=2))]
fn cosine(a: &str, b: &str, n: usize) -> f64 {
    Cosine { n }.similarity(a, b)
}

/// Compute Jaccard similarity between whitespace-tokenized sets.
#[pyfunction]
fn jaccard(a: &str, b: &str) -> f64 {
    Jaccard.similarity(a, b)
}

/// Compute Sorensen-Dice coefficient between character bigrams.
#[pyfunction]
fn sorensen_dice(a: &str, b: &str) -> f64 {
    SorensenDice.similarity(a, b)
}

/// Compute pairwise similarity matrix between two lists of strings.
///
/// Returns a numpy 2D array of shape (len(a), len(b)).
#[pyfunction]
#[pyo3(signature = (a, b, scorer="jaro_winkler", workers=None))]
fn cdist<'py>(
    py: Python<'py>,
    a: Vec<String>,
    b: Vec<String>,
    scorer: &str,
    workers: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let metric =
        metrics::metric_from_name(scorer).map_err(|e| PyValueError::new_err(e.to_string()))?;

    if let Some(n) = workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // Ignore if already initialized
    }

    let rows = a.len();
    let cols = b.len();

    let b_ref = &b;
    let flat: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|i| {
            let metric = &metric;
            let a_str = &a[i];
            (0..cols)
                .map(move |j| metric.similarity(a_str, &b_ref[j]))
                .collect::<Vec<f64>>()
        })
        .collect();

    let array = numpy::PyArray1::from_vec(py, flat)
        .reshape([rows, cols])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(array)
}

// ---------------------------------------------------------------------------
// Phonetic algorithms
// ---------------------------------------------------------------------------

/// Compute the Soundex code for a string.
#[pyfunction]
fn soundex(s: &str) -> String {
    phonetic_mod::Soundex.encode(s)
}

/// Compute the Metaphone code for a string.
#[pyfunction]
fn metaphone(s: &str) -> String {
    phonetic_mod::Metaphone.encode(s)
}

/// Compute the Double Metaphone codes for a string (primary, alternate).
#[pyfunction]
fn double_metaphone(s: &str) -> (String, String) {
    phonetic_mod::DoubleMetaphone.encode_both(s)
}

/// Compute the NYSIIS code for a string.
#[pyfunction]
fn nysiis(s: &str) -> String {
    phonetic_mod::Nysiis.encode(s)
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Fold a string to lowercase.
#[pyfunction]
fn fold_case(s: &str) -> String {
    preprocess::fold_case(s)
}

/// Normalize whitespace (trim + collapse).
#[pyfunction]
fn normalize_whitespace(s: &str) -> String {
    preprocess::normalize_whitespace(s)
}

/// Strip ASCII punctuation from a string.
#[pyfunction]
fn strip_punctuation(s: &str) -> String {
    preprocess::strip_punctuation(s)
}

/// Standardize common name abbreviations.
#[pyfunction]
fn standardize_name(s: &str) -> String {
    preprocess::standardize_name(s)
}

// ---------------------------------------------------------------------------
// EM estimation standalone function
// ---------------------------------------------------------------------------

/// Result of EM estimation of Fellegi-Sunter parameters.
#[pyclass]
#[derive(Debug, Clone)]
struct PyEmResult {
    #[pyo3(get)]
    m_probs: Vec<f64>,
    #[pyo3(get)]
    u_probs: Vec<f64>,
    #[pyo3(get)]
    p_match: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
}

/// Estimate Fellegi-Sunter m/u probabilities from comparison vectors using EM.
#[pyfunction]
#[pyo3(signature = (vectors, max_iterations=100, convergence_threshold=1e-6, initial_p_match=0.1))]
fn estimate_fellegi_sunter_params(
    vectors: Vec<Vec<f64>>,
    max_iterations: usize,
    convergence_threshold: f64,
    initial_p_match: f64,
) -> PyEmResult {
    let config = reclink_core::classify::EmConfig {
        max_iterations,
        convergence_threshold,
        initial_p_match,
    };
    let result = reclink_core::classify::estimate_fellegi_sunter(&vectors, &config);
    PyEmResult {
        m_probs: result.m_probs,
        u_probs: result.u_probs,
        p_match: result.p_match,
        iterations: result.iterations,
        converged: result.converged,
    }
}

// ---------------------------------------------------------------------------
// Pipeline types (exposed for Python pipeline API)
// ---------------------------------------------------------------------------

/// The Python-visible record for pipeline usage.
#[pyclass]
#[derive(Debug, Clone)]
struct PyRecord {
    id: String,
    fields: ahash::AHashMap<String, String>,
}

#[pymethods]
impl PyRecord {
    #[new]
    fn new(id: String) -> Self {
        Self {
            id,
            fields: ahash::AHashMap::new(),
        }
    }

    fn set_field(&mut self, name: String, value: String) {
        self.fields.insert(name, value);
    }

    fn get_field(&self, name: &str) -> Option<String> {
        self.fields.get(name).cloned()
    }
}

/// A match result from the pipeline.
#[pyclass]
#[derive(Debug, Clone)]
struct PyMatchResult {
    #[pyo3(get)]
    left_id: String,
    #[pyo3(get)]
    right_id: String,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    scores: Vec<f64>,
}

/// The main pipeline class exposed to Python.
#[pyclass]
struct PyPipeline {
    blockers: Vec<PyBlockerConfig>,
    comparators: Vec<PyComparatorConfig>,
    classifier: Option<PyClassifierConfig>,
    cluster: PyClusterConfig,
    preprocess_lowercase: Vec<String>,
    preprocess_ops: ahash::AHashMap<String, Vec<String>>,
    numeric_fields: ahash::AHashSet<String>,
    date_fields: ahash::AHashSet<String>,
}

#[derive(Debug, Clone)]
enum PyBlockerConfig {
    Exact {
        field: String,
    },
    Phonetic {
        field: String,
        algorithm: String,
    },
    SortedNeighborhood {
        field: String,
        window: usize,
    },
    Qgram {
        field: String,
        q: usize,
        threshold: usize,
    },
    Lsh {
        field: String,
        num_hashes: usize,
        num_bands: usize,
    },
    Canopy {
        field: String,
        t_tight: f64,
        t_loose: f64,
        metric: String,
    },
    Numeric {
        field: String,
        bucket_size: f64,
    },
    DateBlock {
        field: String,
        resolution: String,
    },
}

#[derive(Debug, Clone)]
enum PyClusterConfig {
    None,
    ConnectedComponents,
    Hierarchical { linkage: String, threshold: f64 },
}

#[derive(Debug, Clone)]
enum PyComparatorConfig {
    String { field: String, metric: String },
    Exact { field: String },
    Numeric { field: String, max_diff: f64 },
    Date { field: String },
    Phonetic { field: String, algorithm: String },
}

#[derive(Debug, Clone)]
enum PyClassifierConfig {
    Threshold {
        threshold: f64,
    },
    Weighted {
        weights: Vec<f64>,
        threshold: f64,
    },
    FellegiSunter {
        m_probs: Vec<f64>,
        u_probs: Vec<f64>,
        upper: f64,
        lower: f64,
    },
    FellegiSunterAuto {
        max_iterations: usize,
        convergence_threshold: f64,
        initial_p_match: f64,
    },
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new() -> Self {
        Self {
            blockers: Vec::new(),
            comparators: Vec::new(),
            classifier: None,
            cluster: PyClusterConfig::None,
            preprocess_lowercase: Vec::new(),
            preprocess_ops: ahash::AHashMap::new(),
            numeric_fields: ahash::AHashSet::new(),
            date_fields: ahash::AHashSet::new(),
        }
    }

    fn preprocess_lowercase(&mut self, fields: Vec<String>) {
        self.preprocess_lowercase = fields;
    }

    fn preprocess(&mut self, field: String, operations: Vec<String>) {
        self.preprocess_ops.insert(field, operations);
    }

    fn block_exact(&mut self, field: String) {
        self.blockers.push(PyBlockerConfig::Exact { field });
    }

    #[pyo3(signature = (field, algorithm="soundex"))]
    fn block_phonetic(&mut self, field: String, algorithm: &str) {
        self.blockers.push(PyBlockerConfig::Phonetic {
            field,
            algorithm: algorithm.to_string(),
        });
    }

    #[pyo3(signature = (field, window=3))]
    fn block_sorted_neighborhood(&mut self, field: String, window: usize) {
        self.blockers
            .push(PyBlockerConfig::SortedNeighborhood { field, window });
    }

    #[pyo3(signature = (field, q=3, threshold=1))]
    fn block_qgram(&mut self, field: String, q: usize, threshold: usize) {
        self.blockers.push(PyBlockerConfig::Qgram {
            field,
            q,
            threshold,
        });
    }

    #[pyo3(signature = (field, metric="jaro_winkler"))]
    fn compare_string(&mut self, field: String, metric: &str) {
        self.comparators.push(PyComparatorConfig::String {
            field,
            metric: metric.to_string(),
        });
    }

    fn compare_exact(&mut self, field: String) {
        self.comparators.push(PyComparatorConfig::Exact { field });
    }

    #[pyo3(signature = (field, max_diff=10.0))]
    fn compare_numeric(&mut self, field: String, max_diff: f64) {
        self.numeric_fields.insert(field.clone());
        self.comparators
            .push(PyComparatorConfig::Numeric { field, max_diff });
    }

    fn compare_date(&mut self, field: String) {
        self.date_fields.insert(field.clone());
        self.comparators.push(PyComparatorConfig::Date { field });
    }

    #[pyo3(signature = (field, algorithm="soundex"))]
    fn compare_phonetic(&mut self, field: String, algorithm: &str) {
        self.comparators.push(PyComparatorConfig::Phonetic {
            field,
            algorithm: algorithm.to_string(),
        });
    }

    fn classify_threshold(&mut self, threshold: f64) {
        self.classifier = Some(PyClassifierConfig::Threshold { threshold });
    }

    fn classify_weighted(&mut self, weights: Vec<f64>, threshold: f64) {
        self.classifier = Some(PyClassifierConfig::Weighted { weights, threshold });
    }

    fn classify_fellegi_sunter(
        &mut self,
        m_probs: Vec<f64>,
        u_probs: Vec<f64>,
        upper: f64,
        lower: f64,
    ) {
        self.classifier = Some(PyClassifierConfig::FellegiSunter {
            m_probs,
            u_probs,
            upper,
            lower,
        });
    }

    #[pyo3(signature = (max_iterations=100, convergence_threshold=1e-6, initial_p_match=0.1))]
    fn classify_fellegi_sunter_auto(
        &mut self,
        max_iterations: usize,
        convergence_threshold: f64,
        initial_p_match: f64,
    ) {
        self.classifier = Some(PyClassifierConfig::FellegiSunterAuto {
            max_iterations,
            convergence_threshold,
            initial_p_match,
        });
    }

    fn cluster_connected_components(&mut self) {
        self.cluster = PyClusterConfig::ConnectedComponents;
    }

    #[pyo3(signature = (linkage="single", threshold=0.5))]
    fn cluster_hierarchical(&mut self, linkage: &str, threshold: f64) {
        self.cluster = PyClusterConfig::Hierarchical {
            linkage: linkage.to_string(),
            threshold,
        };
    }

    #[pyo3(signature = (field, num_hashes=100, num_bands=20))]
    fn block_lsh(&mut self, field: String, num_hashes: usize, num_bands: usize) {
        self.blockers.push(PyBlockerConfig::Lsh {
            field,
            num_hashes,
            num_bands,
        });
    }

    #[pyo3(signature = (field, t_tight=0.9, t_loose=0.5, metric="jaro_winkler"))]
    fn block_canopy(&mut self, field: String, t_tight: f64, t_loose: f64, metric: &str) {
        self.blockers.push(PyBlockerConfig::Canopy {
            field,
            t_tight,
            t_loose,
            metric: metric.to_string(),
        });
    }

    #[pyo3(signature = (field, bucket_size=5.0))]
    fn block_numeric(&mut self, field: String, bucket_size: f64) {
        self.numeric_fields.insert(field.clone());
        self.blockers
            .push(PyBlockerConfig::Numeric { field, bucket_size });
    }

    #[pyo3(signature = (field, resolution="year"))]
    fn block_date(&mut self, field: String, resolution: &str) {
        self.date_fields.insert(field.clone());
        self.blockers.push(PyBlockerConfig::DateBlock {
            field,
            resolution: resolution.to_string(),
        });
    }

    /// Run deduplication on a list of records.
    fn dedup(&self, records: Vec<PyRef<PyRecord>>) -> PyResult<Vec<PyMatchResult>> {
        let batch = self.build_record_batch(&records)?;

        match &self.classifier {
            Some(PyClassifierConfig::FellegiSunterAuto {
                max_iterations,
                convergence_threshold,
                initial_p_match,
            }) => {
                let blockers = self.build_blockers()?;
                let comparators = self.build_comparators()?;

                // Generate candidates and comparison vectors
                let candidates = generate_dedup_candidates(&blockers, &batch);
                let vectors = compare_pairs(&comparators, &batch, &batch, &candidates);

                // Run EM to estimate parameters
                let raw_vectors: Vec<Vec<f64>> = vectors.iter().map(|v| v.scores.clone()).collect();
                let config = reclink_core::classify::EmConfig {
                    max_iterations: *max_iterations,
                    convergence_threshold: *convergence_threshold,
                    initial_p_match: *initial_p_match,
                };
                let em_result =
                    reclink_core::classify::estimate_fellegi_sunter(&raw_vectors, &config);

                // Use EM results to classify
                let classifier = reclink_core::classify::FellegiSunterClassifier::new(
                    em_result.m_probs,
                    em_result.u_probs,
                    4.0,  // upper threshold
                    -4.0, // lower threshold
                );

                let matches: Vec<_> = vectors
                    .into_iter()
                    .map(|v| classifier.classify(&v))
                    .filter(|c| {
                        c.class == reclink_core::record::MatchClass::Match
                            || c.class == reclink_core::record::MatchClass::Possible
                    })
                    .collect();

                Ok(matches
                    .into_iter()
                    .map(|m| PyMatchResult {
                        left_id: batch.records[m.pair.left].id.clone(),
                        right_id: batch.records[m.pair.right].id.clone(),
                        score: m.aggregate_score,
                        scores: m.scores,
                    })
                    .collect())
            }
            _ => {
                let pipeline = self.build_pipeline()?;
                let matches = pipeline.dedup(&batch);
                Ok(matches
                    .into_iter()
                    .map(|m| PyMatchResult {
                        left_id: batch.records[m.pair.left].id.clone(),
                        right_id: batch.records[m.pair.right].id.clone(),
                        score: m.aggregate_score,
                        scores: m.scores,
                    })
                    .collect())
            }
        }
    }

    /// Run dedup and cluster, returning groups of record IDs.
    fn dedup_cluster(&self, records: Vec<PyRef<PyRecord>>) -> PyResult<Vec<Vec<String>>> {
        let batch = self.build_record_batch(&records)?;

        match &self.classifier {
            Some(PyClassifierConfig::FellegiSunterAuto {
                max_iterations,
                convergence_threshold,
                initial_p_match,
            }) => {
                let blockers = self.build_blockers()?;
                let comparators = self.build_comparators()?;

                let candidates = generate_dedup_candidates(&blockers, &batch);
                let vectors = compare_pairs(&comparators, &batch, &batch, &candidates);

                let raw_vectors: Vec<Vec<f64>> = vectors.iter().map(|v| v.scores.clone()).collect();
                let config = reclink_core::classify::EmConfig {
                    max_iterations: *max_iterations,
                    convergence_threshold: *convergence_threshold,
                    initial_p_match: *initial_p_match,
                };
                let em_result =
                    reclink_core::classify::estimate_fellegi_sunter(&raw_vectors, &config);

                let classifier = reclink_core::classify::FellegiSunterClassifier::new(
                    em_result.m_probs,
                    em_result.u_probs,
                    4.0,
                    -4.0,
                );

                let matches: Vec<_> = vectors
                    .into_iter()
                    .map(|v| classifier.classify(&v))
                    .filter(|c| {
                        c.class == reclink_core::record::MatchClass::Match
                            || c.class == reclink_core::record::MatchClass::Possible
                    })
                    .collect();

                let clusters = cluster_matches(&self.cluster, &matches, batch.len())?;
                Ok(clusters
                    .into_iter()
                    .map(|group| {
                        group
                            .into_iter()
                            .map(|i| batch.records[i].id.clone())
                            .collect()
                    })
                    .collect())
            }
            _ => {
                let pipeline = self.build_pipeline()?;
                let clusters = pipeline.dedup_cluster(&batch);
                Ok(clusters
                    .into_iter()
                    .map(|group| {
                        group
                            .into_iter()
                            .map(|i| batch.records[i].id.clone())
                            .collect()
                    })
                    .collect())
            }
        }
    }

    /// Run linkage between two sets of records.
    fn link(
        &self,
        left: Vec<PyRef<PyRecord>>,
        right: Vec<PyRef<PyRecord>>,
    ) -> PyResult<Vec<PyMatchResult>> {
        let left_batch = self.build_record_batch(&left)?;
        let right_batch = self.build_record_batch(&right)?;

        match &self.classifier {
            Some(PyClassifierConfig::FellegiSunterAuto {
                max_iterations,
                convergence_threshold,
                initial_p_match,
            }) => {
                let blockers = self.build_blockers()?;
                let comparators = self.build_comparators()?;

                let candidates = generate_link_candidates(&blockers, &left_batch, &right_batch);
                let vectors = compare_pairs(&comparators, &left_batch, &right_batch, &candidates);

                let raw_vectors: Vec<Vec<f64>> = vectors.iter().map(|v| v.scores.clone()).collect();
                let config = reclink_core::classify::EmConfig {
                    max_iterations: *max_iterations,
                    convergence_threshold: *convergence_threshold,
                    initial_p_match: *initial_p_match,
                };
                let em_result =
                    reclink_core::classify::estimate_fellegi_sunter(&raw_vectors, &config);

                let classifier = reclink_core::classify::FellegiSunterClassifier::new(
                    em_result.m_probs,
                    em_result.u_probs,
                    4.0,
                    -4.0,
                );

                let matches: Vec<_> = vectors
                    .into_iter()
                    .map(|v| classifier.classify(&v))
                    .filter(|c| {
                        c.class == reclink_core::record::MatchClass::Match
                            || c.class == reclink_core::record::MatchClass::Possible
                    })
                    .collect();

                Ok(matches
                    .into_iter()
                    .map(|m| PyMatchResult {
                        left_id: left_batch.records[m.pair.left].id.clone(),
                        right_id: right_batch.records[m.pair.right].id.clone(),
                        score: m.aggregate_score,
                        scores: m.scores,
                    })
                    .collect())
            }
            _ => {
                let pipeline = self.build_pipeline()?;
                let matches = pipeline.link(&left_batch, &right_batch);
                Ok(matches
                    .into_iter()
                    .map(|m| PyMatchResult {
                        left_id: left_batch.records[m.pair.left].id.clone(),
                        right_id: right_batch.records[m.pair.right].id.clone(),
                        score: m.aggregate_score,
                        scores: m.scores,
                    })
                    .collect())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions for EM-auto path
// ---------------------------------------------------------------------------

fn generate_dedup_candidates(
    blockers: &[Box<dyn reclink_core::blocking::BlockingStrategy>],
    records: &reclink_core::record::RecordBatch,
) -> Vec<reclink_core::record::CandidatePair> {
    let all: Vec<Vec<reclink_core::record::CandidatePair>> = blockers
        .par_iter()
        .map(|b| b.block_dedup(records))
        .collect();
    dedup_candidate_pairs(all)
}

fn generate_link_candidates(
    blockers: &[Box<dyn reclink_core::blocking::BlockingStrategy>],
    left: &reclink_core::record::RecordBatch,
    right: &reclink_core::record::RecordBatch,
) -> Vec<reclink_core::record::CandidatePair> {
    let all: Vec<Vec<reclink_core::record::CandidatePair>> = blockers
        .par_iter()
        .map(|b| b.block_link(left, right))
        .collect();
    dedup_candidate_pairs(all)
}

fn compare_pairs(
    comparators: &[Box<dyn reclink_core::compare::FieldComparator>],
    left: &reclink_core::record::RecordBatch,
    right: &reclink_core::record::RecordBatch,
    candidates: &[reclink_core::record::CandidatePair],
) -> Vec<reclink_core::record::ComparisonVector> {
    candidates
        .par_iter()
        .map(|pair| {
            let scores: Vec<f64> = comparators
                .iter()
                .map(|cmp| {
                    let left_val = left.records[pair.left]
                        .get(cmp.field_name())
                        .cloned()
                        .unwrap_or(reclink_core::record::FieldValue::Null);
                    let right_val = right.records[pair.right]
                        .get(cmp.field_name())
                        .cloned()
                        .unwrap_or(reclink_core::record::FieldValue::Null);
                    cmp.compare(&left_val, &right_val)
                })
                .collect();
            reclink_core::record::ComparisonVector {
                pair: *pair,
                scores,
            }
        })
        .collect()
}

fn dedup_candidate_pairs(
    all: Vec<Vec<reclink_core::record::CandidatePair>>,
) -> Vec<reclink_core::record::CandidatePair> {
    let mut seen = ahash::AHashSet::new();
    let mut result = Vec::new();
    for pairs in all {
        for pair in pairs {
            let key = if pair.left <= pair.right {
                (pair.left, pair.right)
            } else {
                (pair.right, pair.left)
            };
            if seen.insert(key) {
                result.push(pair);
            }
        }
    }
    result
}

fn cluster_matches(
    cluster_config: &PyClusterConfig,
    matches: &[reclink_core::record::ClassifiedPair],
    n_records: usize,
) -> PyResult<Vec<Vec<usize>>> {
    use reclink_core::cluster::{ConnectedComponents, HierarchicalClustering};

    match cluster_config {
        PyClusterConfig::None => Ok(matches
            .iter()
            .map(|m| vec![m.pair.left, m.pair.right])
            .collect()),
        PyClusterConfig::ConnectedComponents => {
            let edges: Vec<(usize, usize)> = matches
                .iter()
                .map(|m| (m.pair.left, m.pair.right))
                .collect();
            Ok(ConnectedComponents::find(n_records, &edges))
        }
        PyClusterConfig::Hierarchical { linkage, threshold } => {
            let l = match linkage.as_str() {
                "single" => reclink_core::cluster::Linkage::Single,
                "complete" => reclink_core::cluster::Linkage::Complete,
                "average" => reclink_core::cluster::Linkage::Average,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown linkage: {linkage}. Expected: single, complete, average"
                    )));
                }
            };
            let similarities: Vec<(usize, usize, f64)> = matches
                .iter()
                .map(|m| (m.pair.left, m.pair.right, m.aggregate_score))
                .collect();
            let hc = HierarchicalClustering::new(l, *threshold);
            Ok(hc.cluster(n_records, &similarities))
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline building helpers
// ---------------------------------------------------------------------------

fn parse_preprocess_ops(names: &[String]) -> PyResult<Vec<preprocess::PreprocessOp>> {
    names
        .iter()
        .map(|name| match name.as_str() {
            "fold_case" => Ok(preprocess::PreprocessOp::FoldCase),
            "normalize_whitespace" => Ok(preprocess::PreprocessOp::NormalizeWhitespace),
            "strip_punctuation" => Ok(preprocess::PreprocessOp::StripPunctuation),
            "standardize_name" => Ok(preprocess::PreprocessOp::StandardizeName),
            "normalize_unicode_nfc" => Ok(preprocess::PreprocessOp::NormalizeUnicode(
                preprocess::NormalizationForm::Nfc,
            )),
            "normalize_unicode_nfd" => Ok(preprocess::PreprocessOp::NormalizeUnicode(
                preprocess::NormalizationForm::Nfd,
            )),
            "normalize_unicode_nfkc" => Ok(preprocess::PreprocessOp::NormalizeUnicode(
                preprocess::NormalizationForm::Nfkc,
            )),
            "normalize_unicode_nfkd" => Ok(preprocess::PreprocessOp::NormalizeUnicode(
                preprocess::NormalizationForm::Nfkd,
            )),
            _ => Err(PyValueError::new_err(format!(
                "unknown operation: {name}. Expected: fold_case, normalize_whitespace, \
                 strip_punctuation, standardize_name, normalize_unicode_nfc, \
                 normalize_unicode_nfd, normalize_unicode_nfkc, normalize_unicode_nfkd"
            ))),
        })
        .collect()
}

fn parse_phonetic_algorithm(name: &str) -> PyResult<phonetic_mod::PhoneticAlgorithm> {
    match name {
        "soundex" => Ok(phonetic_mod::PhoneticAlgorithm::Soundex(
            phonetic_mod::Soundex,
        )),
        "metaphone" => Ok(phonetic_mod::PhoneticAlgorithm::Metaphone(
            phonetic_mod::Metaphone,
        )),
        "double_metaphone" => Ok(phonetic_mod::PhoneticAlgorithm::DoubleMetaphone(
            phonetic_mod::DoubleMetaphone,
        )),
        "nysiis" => Ok(phonetic_mod::PhoneticAlgorithm::Nysiis(
            phonetic_mod::Nysiis,
        )),
        _ => Err(PyValueError::new_err(format!(
            "unknown phonetic algorithm: {name}"
        ))),
    }
}

fn parse_date_resolution(name: &str) -> PyResult<reclink_core::blocking::DateResolution> {
    match name {
        "year" => Ok(reclink_core::blocking::DateResolution::Year),
        "month" => Ok(reclink_core::blocking::DateResolution::Month),
        "day" => Ok(reclink_core::blocking::DateResolution::Day),
        _ => Err(PyValueError::new_err(format!(
            "unknown date resolution: {name}. Expected: year, month, day"
        ))),
    }
}

impl PyPipeline {
    fn build_record_batch(
        &self,
        records: &[PyRef<PyRecord>],
    ) -> PyResult<reclink_core::record::RecordBatch> {
        // Pre-parse preprocess ops for each field
        let mut parsed_ops: ahash::AHashMap<String, Vec<preprocess::PreprocessOp>> =
            ahash::AHashMap::new();
        for (field, ops) in &self.preprocess_ops {
            parsed_ops.insert(field.clone(), parse_preprocess_ops(ops)?);
        }

        let mut field_names: Vec<String> = Vec::new();
        if let Some(first) = records.first() {
            field_names = first.fields.keys().cloned().collect();
            field_names.sort();
        }

        let core_records: Vec<reclink_core::record::Record> = records
            .iter()
            .map(|r| {
                let mut rec = reclink_core::record::Record::new(r.id.clone());
                for (k, v) in &r.fields {
                    let field_value = if self.numeric_fields.contains(k) {
                        // Try to parse as numeric
                        if let Ok(f) = v.parse::<f64>() {
                            reclink_core::record::FieldValue::Float(f)
                        } else {
                            reclink_core::record::FieldValue::Text(v.clone())
                        }
                    } else if self.date_fields.contains(k) {
                        reclink_core::record::FieldValue::Date(v.clone())
                    } else {
                        // Apply per-field preprocess ops
                        let mut value = v.clone();
                        if let Some(ops) = parsed_ops.get(k) {
                            value = preprocess::apply_ops(&value, ops);
                        }
                        // Backward compat: preprocess_lowercase
                        if self.preprocess_lowercase.contains(k) {
                            value = value.to_lowercase();
                        }
                        reclink_core::record::FieldValue::Text(value)
                    };
                    rec.fields.insert(k.clone(), field_value);
                }
                rec
            })
            .collect();

        Ok(reclink_core::record::RecordBatch::new(
            field_names,
            core_records,
        ))
    }

    fn build_blockers(&self) -> PyResult<Vec<Box<dyn reclink_core::blocking::BlockingStrategy>>> {
        use reclink_core::blocking::*;

        let mut blockers: Vec<Box<dyn BlockingStrategy>> = Vec::new();
        for blocker_cfg in &self.blockers {
            let blocker: Box<dyn BlockingStrategy> = match blocker_cfg {
                PyBlockerConfig::Exact { field } => Box::new(ExactBlocking::new(field.clone())),
                PyBlockerConfig::Phonetic { field, algorithm } => {
                    let algo = parse_phonetic_algorithm(algorithm)?;
                    Box::new(PhoneticBlocking::new(field.clone(), algo))
                }
                PyBlockerConfig::SortedNeighborhood { field, window } => {
                    Box::new(SortedNeighborhood::new(field.clone(), *window))
                }
                PyBlockerConfig::Qgram {
                    field,
                    q,
                    threshold,
                } => Box::new(QgramBlocking::new(field.clone(), *q, *threshold)),
                PyBlockerConfig::Lsh {
                    field,
                    num_hashes,
                    num_bands,
                } => Box::new(LshBlocking::new(field.clone(), *num_hashes, *num_bands)),
                PyBlockerConfig::Canopy {
                    field,
                    t_tight,
                    t_loose,
                    metric,
                } => {
                    let m = metrics::metric_from_name(metric)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Box::new(CanopyClustering::new(field.clone(), *t_tight, *t_loose, m))
                }
                PyBlockerConfig::Numeric { field, bucket_size } => {
                    Box::new(NumericBlocking::new(field.clone(), *bucket_size))
                }
                PyBlockerConfig::DateBlock { field, resolution } => {
                    let res = parse_date_resolution(resolution)?;
                    Box::new(DateBlocking::new(field.clone(), res))
                }
            };
            blockers.push(blocker);
        }
        Ok(blockers)
    }

    fn build_comparators(&self) -> PyResult<Vec<Box<dyn reclink_core::compare::FieldComparator>>> {
        use reclink_core::compare::*;

        let mut comparators: Vec<Box<dyn FieldComparator>> = Vec::new();
        for comp_cfg in &self.comparators {
            let comparator: Box<dyn FieldComparator> = match comp_cfg {
                PyComparatorConfig::String { field, metric } => {
                    let m = metrics::metric_from_name(metric)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Box::new(StringComparator::new(field.clone(), m))
                }
                PyComparatorConfig::Exact { field } => {
                    Box::new(ExactComparator::new(field.clone()))
                }
                PyComparatorConfig::Numeric { field, max_diff } => {
                    Box::new(NumericComparator::new(field.clone(), *max_diff))
                }
                PyComparatorConfig::Date { field } => Box::new(DateComparator::new(field.clone())),
                PyComparatorConfig::Phonetic { field, algorithm } => {
                    let algo = parse_phonetic_algorithm(algorithm)?;
                    Box::new(PhoneticComparator::new(field.clone(), algo))
                }
            };
            comparators.push(comparator);
        }
        Ok(comparators)
    }

    fn build_pipeline(&self) -> PyResult<reclink_core::pipeline::ReclinkPipeline> {
        use reclink_core::classify::*;

        let mut builder = reclink_core::pipeline::PipelineBuilder::new();

        for blocker in self.build_blockers()? {
            builder = builder.add_blocker(blocker);
        }

        for comparator in self.build_comparators()? {
            builder = builder.add_comparator(comparator);
        }

        if let Some(ref cls_cfg) = self.classifier {
            let classifier: Box<dyn Classifier> = match cls_cfg {
                PyClassifierConfig::Threshold { threshold } => {
                    Box::new(ThresholdClassifier::new(*threshold))
                }
                PyClassifierConfig::Weighted { weights, threshold } => {
                    Box::new(WeightedSumClassifier::new(weights.clone(), *threshold))
                }
                PyClassifierConfig::FellegiSunter {
                    m_probs,
                    u_probs,
                    upper,
                    lower,
                } => Box::new(FellegiSunterClassifier::new(
                    m_probs.clone(),
                    u_probs.clone(),
                    *upper,
                    *lower,
                )),
                PyClassifierConfig::FellegiSunterAuto { .. } => {
                    // Handled in dedup/link methods directly
                    unreachable!("FellegiSunterAuto should be handled before build_pipeline")
                }
            };
            builder = builder.set_classifier(classifier);
        }

        match &self.cluster {
            PyClusterConfig::None => {}
            PyClusterConfig::ConnectedComponents => {
                builder = builder.with_clustering();
            }
            PyClusterConfig::Hierarchical { linkage, threshold } => {
                let l = match linkage.as_str() {
                    "single" => reclink_core::cluster::Linkage::Single,
                    "complete" => reclink_core::cluster::Linkage::Complete,
                    "average" => reclink_core::cluster::Linkage::Average,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "unknown linkage: {linkage}. Expected: single, complete, average"
                        )));
                    }
                };
                builder = builder.with_hierarchical_clustering(l, *threshold);
            }
        }

        builder
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/// Generate character n-grams from a string.
#[pyfunction]
#[pyo3(signature = (s, n=2))]
fn ngram_tokenize(s: &str, n: usize) -> Vec<String> {
    preprocess::ngram_tokenize(s, n)
}

/// Split a string on whitespace boundaries.
#[pyfunction]
fn whitespace_tokenize(s: &str) -> Vec<String> {
    preprocess::whitespace_tokenize(s)
        .into_iter()
        .map(String::from)
        .collect()
}

/// Apply Unicode normalization to a string.
#[pyfunction]
#[pyo3(signature = (s, form="nfkc"))]
fn normalize_unicode(s: &str, form: &str) -> PyResult<String> {
    let nf = match form {
        "nfc" => preprocess::NormalizationForm::Nfc,
        "nfd" => preprocess::NormalizationForm::Nfd,
        "nfkc" => preprocess::NormalizationForm::Nfkc,
        "nfkd" => preprocess::NormalizationForm::Nfkd,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown normalization form: {form}. Expected: nfc, nfd, nfkc, nfkd"
            )));
        }
    };
    Ok(preprocess::normalize_unicode(s, nf))
}

// ---------------------------------------------------------------------------
// Batch preprocessing
// ---------------------------------------------------------------------------

/// Apply a chain of preprocessing operations to a batch of strings in parallel.
#[pyfunction]
fn preprocess_batch(strings: Vec<String>, operations: Vec<String>) -> PyResult<Vec<String>> {
    let ops = parse_preprocess_ops(&operations)?;
    Ok(preprocess::preprocess_batch(&strings, &ops))
}

/// Generate character n-grams for a batch of strings in parallel.
#[pyfunction]
#[pyo3(signature = (strings, n=2))]
fn ngram_tokenize_batch(strings: Vec<String>, n: usize) -> Vec<Vec<String>> {
    strings
        .par_iter()
        .map(|s| preprocess::ngram_tokenize(s, n))
        .collect()
}

/// Split each string on whitespace for a batch of strings in parallel.
#[pyfunction]
fn whitespace_tokenize_batch(strings: Vec<String>) -> Vec<Vec<String>> {
    strings
        .par_iter()
        .map(|s| {
            preprocess::whitespace_tokenize(s)
                .into_iter()
                .map(String::from)
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // String metrics
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler, m)?)?;
    m.add_function(wrap_pyfunction!(cosine, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(sorensen_dice, m)?)?;
    m.add_function(wrap_pyfunction!(cdist, m)?)?;

    // Phonetic
    m.add_function(wrap_pyfunction!(soundex, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(double_metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(nysiis, m)?)?;

    // Preprocessing
    m.add_function(wrap_pyfunction!(fold_case, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_whitespace, m)?)?;
    m.add_function(wrap_pyfunction!(strip_punctuation, m)?)?;
    m.add_function(wrap_pyfunction!(standardize_name, m)?)?;

    // Tokenization & Unicode normalization
    m.add_function(wrap_pyfunction!(ngram_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(whitespace_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_unicode, m)?)?;

    // Batch preprocessing
    m.add_function(wrap_pyfunction!(preprocess_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(whitespace_tokenize_batch, m)?)?;

    // EM estimation
    m.add_function(wrap_pyfunction!(estimate_fellegi_sunter_params, m)?)?;
    m.add_class::<PyEmResult>()?;

    // Pipeline classes
    m.add_class::<PyRecord>()?;
    m.add_class::<PyMatchResult>()?;
    m.add_class::<PyPipeline>()?;

    Ok(())
}
