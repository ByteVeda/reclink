pub mod builders;
pub mod config;
pub mod helpers;

use pyo3::prelude::*;
use reclink_core::classify::Classifier;

use config::{PyBlockerConfig, PyClassifierConfig, PyClusterConfig, PyComparatorConfig};
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
pub struct PyPipeline {
    pub blockers: Vec<PyBlockerConfig>,
    pub comparators: Vec<PyComparatorConfig>,
    pub classifier: Option<PyClassifierConfig>,
    pub cluster: PyClusterConfig,
    pub preprocess_lowercase: Vec<String>,
    pub preprocess_ops: ahash::AHashMap<String, Vec<String>>,
    pub numeric_fields: ahash::AHashSet<String>,
    pub date_fields: ahash::AHashSet<String>,
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_fellegi_sunter_params, m)?)?;
    m.add_class::<PyEmResult>()?;
    m.add_class::<PyRecord>()?;
    m.add_class::<PyMatchResult>()?;
    m.add_class::<PyPipeline>()?;
    Ok(())
}
