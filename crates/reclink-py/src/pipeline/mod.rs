pub mod builders;
pub mod config;
pub mod helpers;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use pyo3::prelude::*;
use reclink_core::classify::Classifier;
use reclink_core::record::MatchClass;

use config::{
    PipelineConfig, PyBlockerConfig, PyClassifierConfig, PyClusterConfig, PyComparatorConfig,
};
use helpers::{
    cluster_matches, compare_pairs, generate_dedup_candidates, generate_link_candidates,
};

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

/// The Python-visible record for pipeline usage.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyRecord {
    pub id: String,
    pub fields: ahash::AHashMap<String, String>,
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

fn match_class_str(class: MatchClass) -> String {
    match class {
        MatchClass::Match => "match".to_string(),
        MatchClass::Possible => "possible".to_string(),
        MatchClass::NonMatch => "non_match".to_string(),
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
    #[pyo3(get)]
    match_class: String,
}

#[pymethods]
impl PyMatchResult {
    fn __repr__(&self) -> String {
        format!(
            "MatchResult(left_id='{}', right_id='{}', score={:.4}, match_class='{}', scores={:?})",
            self.left_id, self.right_id, self.score, self.match_class, self.scores
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __eq__(&self, other: &PyMatchResult) -> bool {
        self.left_id == other.left_id
            && self.right_id == other.right_id
            && (self.score - other.score).abs() < 1e-10
            && self.scores.len() == other.scores.len()
            && self
                .scores
                .iter()
                .zip(other.scores.iter())
                .all(|(a, b)| (a - b).abs() < 1e-10)
            && self.match_class == other.match_class
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.left_id.hash(&mut hasher);
        self.right_id.hash(&mut hasher);
        self.score.to_bits().hash(&mut hasher);
        self.match_class.hash(&mut hasher);
        hasher.finish()
    }
}

/// The main pipeline class exposed to Python.
#[pyclass]
pub struct PyPipeline {
    pub blockers: Vec<PyBlockerConfig>,
    pub comparators: Vec<PyComparatorConfig>,
    pub classifier: Option<PyClassifierConfig>,
    pub cluster: PyClusterConfig,
    pub preprocess_lowercase: Vec<String>,
    pub preprocess_ops: ahash::AHashMap<String, Vec<String>>,
    pub numeric_fields: ahash::AHashSet<String>,
    pub date_fields: ahash::AHashSet<String>,
    profiling: bool,
    profiling_stats: ahash::AHashMap<String, u64>,
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
            profiling: false,
            profiling_stats: ahash::AHashMap::new(),
        }
    }

    /// Enable or disable profiling.
    fn with_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
    }

    /// Get the profiling stats from the last run.
    ///
    /// Returns a dict mapping stage names to elapsed nanoseconds.
    fn get_profiling_stats(&self) -> std::collections::HashMap<String, u64> {
        self.profiling_stats
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
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

    fn block_custom(&mut self, name: String) {
        self.blockers.push(PyBlockerConfig::Custom { name });
    }

    fn compare_custom(&mut self, field: String, name: String) {
        self.comparators
            .push(PyComparatorConfig::Custom { field, name });
    }

    fn classify_custom(&mut self, name: String) {
        self.classifier = Some(PyClassifierConfig::Custom { name });
    }

    fn classify_threshold(&mut self, threshold: f64) {
        self.classifier = Some(PyClassifierConfig::Threshold { threshold });
    }

    fn classify_weighted(&mut self, weights: Vec<f64>, threshold: f64) {
        self.classifier = Some(PyClassifierConfig::Weighted { weights, threshold });
    }

    fn classify_threshold_bands(&mut self, upper: f64, lower: f64) {
        self.classifier = Some(PyClassifierConfig::ThresholdBands { upper, lower });
    }

    fn classify_weighted_bands(&mut self, weights: Vec<f64>, upper: f64, lower: f64) {
        self.classifier = Some(PyClassifierConfig::WeightedBands {
            weights,
            upper,
            lower,
        });
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

    #[pyo3(signature = (field, min_prefix_len=2, max_frequency=100))]
    fn block_trie(&mut self, field: String, min_prefix_len: usize, max_frequency: usize) {
        self.blockers.push(PyBlockerConfig::Trie {
            field,
            min_prefix_len,
            max_frequency,
        });
    }

    #[pyo3(signature = (field, resolution="year"))]
    fn block_date(&mut self, field: String, resolution: &str) {
        self.date_fields.insert(field.clone());
        self.blockers.push(PyBlockerConfig::DateBlock {
            field,
            resolution: resolution.to_string(),
        });
    }

    /// Serialize the pipeline configuration to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        let mut preprocess_ops = std::collections::BTreeMap::new();
        for (k, v) in &self.preprocess_ops {
            preprocess_ops.insert(k.clone(), v.clone());
        }
        let mut numeric_fields: Vec<String> = self.numeric_fields.iter().cloned().collect();
        numeric_fields.sort();
        let mut date_fields: Vec<String> = self.date_fields.iter().cloned().collect();
        date_fields.sort();

        let config = PipelineConfig {
            blockers: self.blockers.clone(),
            comparators: self.comparators.clone(),
            classifier: self.classifier.clone(),
            cluster: self.cluster.clone(),
            preprocess_lowercase: self.preprocess_lowercase.clone(),
            preprocess_ops,
            numeric_fields,
            date_fields,
        };
        serde_json::to_string_pretty(&config)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Deserialize a pipeline from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let config: PipelineConfig = serde_json::from_str(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            blockers: config.blockers,
            comparators: config.comparators,
            classifier: config.classifier,
            cluster: config.cluster,
            preprocess_lowercase: config.preprocess_lowercase,
            preprocess_ops: config.preprocess_ops.into_iter().collect(),
            numeric_fields: config.numeric_fields.into_iter().collect(),
            date_fields: config.date_fields.into_iter().collect(),
            profiling: false,
            profiling_stats: ahash::AHashMap::new(),
        })
    }

    /// Run deduplication on a list of records.
    fn dedup(
        &mut self,
        py: Python<'_>,
        records: Vec<PyRef<PyRecord>>,
    ) -> PyResult<Vec<PyMatchResult>> {
        self.profiling_stats.clear();
        let profiling = self.profiling;

        let t0 = Instant::now();
        let batch = self.build_record_batch(&records)?;
        if profiling {
            self.profiling_stats
                .insert("preprocess".into(), t0.elapsed().as_nanos() as u64);
        }

        match &self.classifier {
            Some(PyClassifierConfig::FellegiSunterAuto {
                max_iterations,
                convergence_threshold,
                initial_p_match,
            }) => {
                let max_iterations = *max_iterations;
                let convergence_threshold = *convergence_threshold;
                let initial_p_match = *initial_p_match;

                let blockers = self.build_blockers()?;
                let comparators = self.build_comparators()?;

                let (blocking_ns, comparison_ns, classification_ns, matches) =
                    py.allow_threads(|| {
                        let t1 = Instant::now();
                        let candidates = generate_dedup_candidates(&blockers, &batch);
                        let blocking_ns = t1.elapsed().as_nanos() as u64;

                        let t2 = Instant::now();
                        let vectors = compare_pairs(&comparators, &batch, &batch, &candidates);
                        let comparison_ns = t2.elapsed().as_nanos() as u64;

                        let t3 = Instant::now();
                        let raw_vectors: Vec<Vec<f64>> =
                            vectors.iter().map(|v| v.scores.clone()).collect();
                        let config = reclink_core::classify::EmConfig {
                            max_iterations,
                            convergence_threshold,
                            initial_p_match,
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
                        let classification_ns = t3.elapsed().as_nanos() as u64;

                        (blocking_ns, comparison_ns, classification_ns, matches)
                    });

                if profiling {
                    self.profiling_stats.insert("blocking".into(), blocking_ns);
                    self.profiling_stats
                        .insert("comparison".into(), comparison_ns);
                    self.profiling_stats
                        .insert("classification".into(), classification_ns);
                }

                Ok(matches
                    .into_iter()
                    .map(|m| PyMatchResult {
                        left_id: batch.records[m.pair.left].id.clone(),
                        right_id: batch.records[m.pair.right].id.clone(),
                        score: m.aggregate_score,
                        match_class: match_class_str(m.class),
                        scores: m.scores,
                    })
                    .collect())
            }
            _ => {
                let t1 = Instant::now();
                let pipeline = self.build_pipeline()?;
                if profiling {
                    self.profiling_stats
                        .insert("build".into(), t1.elapsed().as_nanos() as u64);
                }

                let t2 = Instant::now();
                let matches = py.allow_threads(|| pipeline.dedup(&batch));
                if profiling {
                    self.profiling_stats
                        .insert("dedup_total".into(), t2.elapsed().as_nanos() as u64);
                }

                Ok(matches
                    .into_iter()
                    .map(|m| PyMatchResult {
                        left_id: batch.records[m.pair.left].id.clone(),
                        right_id: batch.records[m.pair.right].id.clone(),
                        score: m.aggregate_score,
                        match_class: match_class_str(m.class),
                        scores: m.scores,
                    })
                    .collect())
            }
        }
    }

    /// Run dedup and cluster, returning groups of record IDs.
    fn dedup_cluster(
        &mut self,
        py: Python<'_>,
        records: Vec<PyRef<PyRecord>>,
    ) -> PyResult<Vec<Vec<String>>> {
        let batch = self.build_record_batch(&records)?;

        match &self.classifier {
            Some(PyClassifierConfig::FellegiSunterAuto {
                max_iterations,
                convergence_threshold,
                initial_p_match,
            }) => {
                let max_iterations = *max_iterations;
                let convergence_threshold = *convergence_threshold;
                let initial_p_match = *initial_p_match;

                let blockers = self.build_blockers()?;
                let comparators = self.build_comparators()?;
                let cluster_config = self.cluster.clone();

                let clusters = py.allow_threads(|| {
                    let candidates = generate_dedup_candidates(&blockers, &batch);
                    let vectors = compare_pairs(&comparators, &batch, &batch, &candidates);

                    let raw_vectors: Vec<Vec<f64>> =
                        vectors.iter().map(|v| v.scores.clone()).collect();
                    let config = reclink_core::classify::EmConfig {
                        max_iterations,
                        convergence_threshold,
                        initial_p_match,
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

                    cluster_matches(&cluster_config, &matches, batch.len())
                })?;

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
                let clusters = py.allow_threads(|| pipeline.dedup_cluster(&batch));
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
        &mut self,
        py: Python<'_>,
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
                let max_iterations = *max_iterations;
                let convergence_threshold = *convergence_threshold;
                let initial_p_match = *initial_p_match;

                let blockers = self.build_blockers()?;
                let comparators = self.build_comparators()?;

                let matches = py.allow_threads(|| {
                    let candidates = generate_link_candidates(&blockers, &left_batch, &right_batch);
                    let vectors =
                        compare_pairs(&comparators, &left_batch, &right_batch, &candidates);

                    let raw_vectors: Vec<Vec<f64>> =
                        vectors.iter().map(|v| v.scores.clone()).collect();
                    let config = reclink_core::classify::EmConfig {
                        max_iterations,
                        convergence_threshold,
                        initial_p_match,
                    };
                    let em_result =
                        reclink_core::classify::estimate_fellegi_sunter(&raw_vectors, &config);

                    let classifier = reclink_core::classify::FellegiSunterClassifier::new(
                        em_result.m_probs,
                        em_result.u_probs,
                        4.0,
                        -4.0,
                    );

                    vectors
                        .into_iter()
                        .map(|v| classifier.classify(&v))
                        .filter(|c| {
                            c.class == reclink_core::record::MatchClass::Match
                                || c.class == reclink_core::record::MatchClass::Possible
                        })
                        .collect::<Vec<_>>()
                });

                Ok(matches
                    .into_iter()
                    .map(|m| PyMatchResult {
                        left_id: left_batch.records[m.pair.left].id.clone(),
                        right_id: right_batch.records[m.pair.right].id.clone(),
                        score: m.aggregate_score,
                        match_class: match_class_str(m.class),
                        scores: m.scores,
                    })
                    .collect())
            }
            _ => {
                let pipeline = self.build_pipeline()?;
                let matches = py.allow_threads(|| pipeline.link(&left_batch, &right_batch));
                Ok(matches
                    .into_iter()
                    .map(|m| PyMatchResult {
                        left_id: left_batch.records[m.pair.left].id.clone(),
                        right_id: right_batch.records[m.pair.right].id.clone(),
                        score: m.aggregate_score,
                        match_class: match_class_str(m.class),
                        scores: m.scores,
                    })
                    .collect())
            }
        }
    }
}

/// Incremental (streaming) clusterer.
///
/// Assigns records to clusters one at a time. Each cluster has a representative
/// string, and new records join the most similar cluster (if above threshold)
/// or start a new cluster.
#[pyclass]
pub struct PyIncrementalCluster {
    inner: reclink_core::cluster::IncrementalCluster,
}

#[pymethods]
impl PyIncrementalCluster {
    /// Create a new incremental clusterer.
    ///
    /// Parameters
    /// ----------
    /// metric : str
    ///     Metric name (default "jaro_winkler").
    /// threshold : float
    ///     Minimum similarity to join an existing cluster (default 0.85).
    #[new]
    #[pyo3(signature = (metric="jaro_winkler", threshold=0.85))]
    fn new(metric: &str, threshold: f64) -> PyResult<Self> {
        let m = reclink_core::metrics::metric_from_name(metric)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: reclink_core::cluster::IncrementalCluster::new(m, threshold),
        })
    }

    /// Add a record and assign it to a cluster.
    ///
    /// Returns a dict with ``cluster_id`` and either ``similarity`` (existing
    /// cluster) or ``new`` (True if a new cluster was created).
    fn add_record(&mut self, value: &str) -> (usize, bool, Option<f64>) {
        let assignment = self.inner.add_record(value);
        match assignment {
            reclink_core::cluster::ClusterAssignment::Existing {
                cluster_id,
                similarity,
            } => (cluster_id, false, Some(similarity)),
            reclink_core::cluster::ClusterAssignment::New { cluster_id } => {
                (cluster_id, true, None)
            }
        }
    }

    /// Return all clusters as lists of record indices.
    fn get_clusters(&self) -> Vec<Vec<usize>> {
        self.inner.get_clusters().to_vec()
    }

    /// Return the number of clusters.
    fn cluster_count(&self) -> usize {
        self.inner.cluster_count()
    }

    /// Return the number of records added.
    fn record_count(&self) -> usize {
        self.inner.record_count()
    }
}

/// Compute silhouette score for a clustering.
#[pyfunction]
fn silhouette_score(
    num_nodes: usize,
    similarities: Vec<(usize, usize, f64)>,
    labels: Vec<i32>,
) -> f64 {
    reclink_core::cluster::silhouette_score(num_nodes, &similarities, &labels)
}

/// Compute Davies-Bouldin index for a clustering.
#[pyfunction]
fn davies_bouldin_index(
    num_nodes: usize,
    similarities: Vec<(usize, usize, f64)>,
    labels: Vec<i32>,
) -> f64 {
    reclink_core::cluster::davies_bouldin_index(num_nodes, &similarities, &labels)
}

/// Train a logistic regression classifier from labeled data.
#[pyfunction]
#[pyo3(signature = (vectors, labels, learning_rate=0.1, max_iterations=1000, regularization=0.01, threshold=0.5))]
fn train_logistic_regression(
    vectors: Vec<Vec<f64>>,
    labels: Vec<bool>,
    learning_rate: f64,
    max_iterations: usize,
    regularization: f64,
    threshold: f64,
) -> (Vec<f64>, f64, f64) {
    let config = reclink_core::classify::LogisticRegressionConfig {
        learning_rate,
        max_iterations,
        convergence_threshold: 1e-6,
        regularization,
        threshold,
    };
    let clf = reclink_core::classify::train_logistic_regression(&vectors, &labels, &config);
    (clf.weights, clf.bias, clf.threshold)
}

/// Train a decision tree classifier from labeled data.
#[pyfunction]
#[pyo3(signature = (vectors, labels, max_depth=5, min_samples_leaf=5, min_samples_split=10, match_threshold=0.5))]
fn train_decision_tree(
    vectors: Vec<Vec<f64>>,
    labels: Vec<bool>,
    max_depth: usize,
    min_samples_leaf: usize,
    min_samples_split: usize,
    match_threshold: f64,
) -> PyResult<String> {
    let config = reclink_core::classify::DecisionTreeConfig {
        max_depth,
        min_samples_leaf,
        min_samples_split,
        match_threshold,
    };
    let clf = reclink_core::classify::train_decision_tree(&vectors, &labels, &config);
    serde_json::to_string(&clf).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Run DBSCAN clustering on pairwise similarities.
#[pyfunction]
fn dbscan_cluster(
    num_nodes: usize,
    similarities: Vec<(usize, usize, f64)>,
    min_similarity: f64,
    min_samples: usize,
) -> (Vec<Vec<usize>>, Vec<usize>, Vec<i32>) {
    let db = reclink_core::cluster::Dbscan::new(min_similarity, min_samples);
    let result = db.cluster(num_nodes, &similarities);
    (result.clusters, result.noise, result.labels)
}

/// Run OPTICS clustering on pairwise similarities.
#[pyfunction]
fn optics_cluster(
    num_nodes: usize,
    similarities: Vec<(usize, usize, f64)>,
    min_samples: usize,
    extract_threshold: f64,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let optics = reclink_core::cluster::Optics::new(min_samples, extract_threshold);
    let result = optics.cluster(num_nodes, &similarities);
    (result.clusters, result.noise)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_fellegi_sunter_params, m)?)?;
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_index, m)?)?;
    m.add_function(wrap_pyfunction!(train_logistic_regression, m)?)?;
    m.add_function(wrap_pyfunction!(train_decision_tree, m)?)?;
    m.add_function(wrap_pyfunction!(dbscan_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(optics_cluster, m)?)?;
    m.add_class::<PyEmResult>()?;
    m.add_class::<PyRecord>()?;
    m.add_class::<PyMatchResult>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyIncrementalCluster>()?;
    Ok(())
}
