//! Builder-pattern pipeline orchestrator for end-to-end record linkage.
//!
//! The pipeline combines preprocessing, blocking, comparison, classification,
//! and clustering into a single configurable workflow.

use crate::blocking::BlockingStrategy;
use crate::classify::Classifier;
use crate::cluster::{ConnectedComponents, HierarchicalClustering, Linkage};
use crate::compare::FieldComparator;
use crate::error::{ReclinkError, Result};
use crate::record::{CandidatePair, ClassifiedPair, ComparisonVector, MatchClass, RecordBatch};
use rayon::prelude::*;

/// Configuration for how matched pairs are clustered.
#[derive(Debug, Clone, Default)]
pub enum ClusterConfig {
    /// No clustering — return raw pair-wise groups.
    #[default]
    None,
    /// Union-find connected components.
    ConnectedComponents,
    /// Hierarchical agglomerative clustering.
    Hierarchical {
        /// Linkage criterion (single, complete, average).
        linkage: Linkage,
        /// Distance threshold for merging clusters.
        threshold: f64,
    },
}

/// A configured record linkage pipeline.
pub struct ReclinkPipeline {
    blockers: Vec<Box<dyn BlockingStrategy>>,
    comparators: Vec<Box<dyn FieldComparator>>,
    classifier: Box<dyn Classifier>,
    cluster: ClusterConfig,
}

impl ReclinkPipeline {
    /// Creates a new [`PipelineBuilder`].
    #[must_use]
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Deduplicates a single dataset, returning groups of matching record indices.
    #[must_use]
    pub fn dedup(&self, records: &RecordBatch) -> Vec<ClassifiedPair> {
        let candidates = self.generate_dedup_candidates(records);
        let vectors = self.compare_pairs(records, records, &candidates);
        let classified: Vec<ClassifiedPair> = vectors
            .into_iter()
            .map(|v| self.classifier.classify(&v))
            .filter(|c| c.class == MatchClass::Match || c.class == MatchClass::Possible)
            .collect();
        classified
    }

    /// Deduplicates and clusters, returning groups of record indices.
    #[must_use]
    pub fn dedup_cluster(&self, records: &RecordBatch) -> Vec<Vec<usize>> {
        let matches = self.dedup(records);
        match &self.cluster {
            ClusterConfig::None => matches
                .into_iter()
                .map(|m| vec![m.pair.left, m.pair.right])
                .collect(),
            ClusterConfig::ConnectedComponents => {
                let edges: Vec<(usize, usize)> = matches
                    .iter()
                    .map(|m| (m.pair.left, m.pair.right))
                    .collect();
                ConnectedComponents::find(records.len(), &edges)
            }
            ClusterConfig::Hierarchical { linkage, threshold } => {
                let similarities: Vec<(usize, usize, f64)> = matches
                    .iter()
                    .map(|m| (m.pair.left, m.pair.right, m.aggregate_score))
                    .collect();
                let hc = HierarchicalClustering::new(*linkage, *threshold);
                hc.cluster(records.len(), &similarities)
            }
        }
    }

    /// Links two datasets, returning matching pairs.
    #[must_use]
    pub fn link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<ClassifiedPair> {
        let candidates = self.generate_link_candidates(left, right);
        let vectors = self.compare_pairs(left, right, &candidates);
        vectors
            .into_iter()
            .map(|v| self.classifier.classify(&v))
            .filter(|c| c.class == MatchClass::Match || c.class == MatchClass::Possible)
            .collect()
    }

    fn generate_dedup_candidates(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let all: Vec<Vec<CandidatePair>> = self
            .blockers
            .par_iter()
            .map(|b| b.block_dedup(records))
            .collect();
        dedup_pairs(all)
    }

    fn generate_link_candidates(
        &self,
        left: &RecordBatch,
        right: &RecordBatch,
    ) -> Vec<CandidatePair> {
        let all: Vec<Vec<CandidatePair>> = self
            .blockers
            .par_iter()
            .map(|b| b.block_link(left, right))
            .collect();
        dedup_pairs(all)
    }

    fn compare_pairs(
        &self,
        left: &RecordBatch,
        right: &RecordBatch,
        candidates: &[CandidatePair],
    ) -> Vec<ComparisonVector> {
        candidates
            .par_iter()
            .map(|pair| {
                let scores: Vec<f64> = self
                    .comparators
                    .iter()
                    .map(|cmp| {
                        let left_val = left.records[pair.left]
                            .get(cmp.field_name())
                            .cloned()
                            .unwrap_or(crate::record::FieldValue::Null);
                        let right_val = right.records[pair.right]
                            .get(cmp.field_name())
                            .cloned()
                            .unwrap_or(crate::record::FieldValue::Null);
                        cmp.compare(&left_val, &right_val)
                    })
                    .collect();
                ComparisonVector {
                    pair: *pair,
                    scores,
                }
            })
            .collect()
    }
}

/// Builder for constructing a [`ReclinkPipeline`].
pub struct PipelineBuilder {
    blockers: Vec<Box<dyn BlockingStrategy>>,
    comparators: Vec<Box<dyn FieldComparator>>,
    classifier: Option<Box<dyn Classifier>>,
    cluster: ClusterConfig,
}

impl PipelineBuilder {
    /// Creates a new empty pipeline builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            blockers: Vec::new(),
            comparators: Vec::new(),
            classifier: None,
            cluster: ClusterConfig::None,
        }
    }

    /// Adds a blocking strategy.
    pub fn add_blocker(mut self, blocker: Box<dyn BlockingStrategy>) -> Self {
        self.blockers.push(blocker);
        self
    }

    /// Adds a field comparator.
    pub fn add_comparator(mut self, comparator: Box<dyn FieldComparator>) -> Self {
        self.comparators.push(comparator);
        self
    }

    /// Sets the classifier.
    pub fn set_classifier(mut self, classifier: Box<dyn Classifier>) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Enables connected-component clustering of results.
    pub fn with_clustering(mut self) -> Self {
        self.cluster = ClusterConfig::ConnectedComponents;
        self
    }

    /// Enables hierarchical agglomerative clustering of results.
    pub fn with_hierarchical_clustering(mut self, linkage: Linkage, threshold: f64) -> Self {
        self.cluster = ClusterConfig::Hierarchical { linkage, threshold };
        self
    }

    /// Builds the pipeline, returning an error if not fully configured.
    pub fn build(self) -> Result<ReclinkPipeline> {
        if self.blockers.is_empty() {
            return Err(ReclinkError::Pipeline(
                "at least one blocking strategy is required".into(),
            ));
        }
        if self.comparators.is_empty() {
            return Err(ReclinkError::Pipeline(
                "at least one field comparator is required".into(),
            ));
        }
        let classifier = self
            .classifier
            .ok_or_else(|| ReclinkError::Pipeline("a classifier is required".into()))?;
        Ok(ReclinkPipeline {
            blockers: self.blockers,
            comparators: self.comparators,
            classifier,
            cluster: self.cluster,
        })
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Deduplicates candidate pairs from multiple blockers.
fn dedup_pairs(all: Vec<Vec<CandidatePair>>) -> Vec<CandidatePair> {
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
