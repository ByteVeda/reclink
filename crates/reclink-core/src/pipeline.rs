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
        let n = self.comparators.len();
        candidates
            .par_iter()
            .map(|pair| {
                let mut scores = vec![0.0; n];
                for (i, cmp) in self.comparators.iter().enumerate() {
                    let left_val = left.records[pair.left]
                        .get(cmp.field_name())
                        .cloned()
                        .unwrap_or(crate::record::FieldValue::Null);
                    let right_val = right.records[pair.right]
                        .get(cmp.field_name())
                        .cloned()
                        .unwrap_or(crate::record::FieldValue::Null);
                    scores[i] = cmp.compare(&left_val, &right_val);
                    if self.classifier.can_reject_early(&scores, n) {
                        break; // remaining scores stay 0.0
                    }
                }
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
    selectivity_overrides: Option<Vec<f64>>,
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
            selectivity_overrides: None,
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

    /// Overrides selectivity hints for each comparator (in insertion order).
    ///
    /// The length must match the number of comparators added so far.
    /// Higher values mean more selective (the pipeline sorts by
    /// `estimated_cost / selectivity` to run cheap, selective comparators first).
    pub fn with_selectivity_hints(mut self, hints: Vec<f64>) -> Self {
        self.selectivity_overrides = Some(hints);
        self
    }

    /// Builds the pipeline, returning an error if not fully configured.
    ///
    /// Comparators are stable-sorted by `estimated_cost / selectivity_hint`
    /// so the pipeline evaluates cheap, high-selectivity comparators first,
    /// enabling early termination via [`Classifier::can_reject_early`].
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
        if let Some(ref hints) = self.selectivity_overrides {
            if hints.len() != self.comparators.len() {
                return Err(ReclinkError::Pipeline(format!(
                    "selectivity_hints length ({}) must match comparators length ({})",
                    hints.len(),
                    self.comparators.len()
                )));
            }
        }
        let classifier = self
            .classifier
            .ok_or_else(|| ReclinkError::Pipeline("a classifier is required".into()))?;

        // Pair comparators with their index (for selectivity overrides)
        let mut indexed: Vec<(usize, Box<dyn FieldComparator>)> =
            self.comparators.into_iter().enumerate().collect();

        let overrides = &self.selectivity_overrides;
        indexed.sort_by(|(i_a, a), (i_b, b)| {
            let sel_a = overrides
                .as_ref()
                .map_or_else(|| a.selectivity_hint(), |h| h[*i_a]);
            let sel_b = overrides
                .as_ref()
                .map_or_else(|| b.selectivity_hint(), |h| h[*i_b]);
            let key_a = a.estimated_cost() as f64 / sel_a;
            let key_b = b.estimated_cost() as f64 / sel_b;
            key_a
                .partial_cmp(&key_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let comparators: Vec<Box<dyn FieldComparator>> =
            indexed.into_iter().map(|(_, c)| c).collect();

        Ok(ReclinkPipeline {
            blockers: self.blockers,
            comparators,
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

/// Helper to compute the composite sort key for a comparator.
#[cfg(test)]
fn sort_key(cmp: &dyn FieldComparator) -> f64 {
    cmp.estimated_cost() as f64 / cmp.selectivity_hint()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{DateComparator, ExactComparator, NumericComparator, StringComparator};
    use crate::metrics::Metric;

    #[test]
    fn composite_key_ordering() {
        // Exact: cost=1, sel=5.0, key=0.2
        // Numeric: cost=2, sel=3.0, key=0.67
        // Date: cost=5, sel=4.0, key=1.25
        // String(Jaro): cost=20, sel=1.5, key=13.3
        let exact = ExactComparator::new("id");
        let numeric = NumericComparator::new("age", 10.0);
        let date = DateComparator::new("dob");
        let string = StringComparator::new("name", Metric::default());

        let key_exact = sort_key(&exact);
        let key_numeric = sort_key(&numeric);
        let key_date = sort_key(&date);
        let key_string = sort_key(&string);

        assert!(key_exact < key_numeric);
        assert!(key_numeric < key_date);
        assert!(key_date < key_string);
    }

    #[test]
    fn selectivity_overrides_change_order() {
        // Without overrides: exact (key=0.2) comes before string (key=13.3)
        // With overrides: give string selectivity=100.0, key=20/100=0.2
        //                 give exact selectivity=0.1, key=1/0.1=10.0
        // So string should come first
        let builder = ReclinkPipeline::builder()
            .add_blocker(Box::new(crate::blocking::ExactBlocking::new("id")))
            .add_comparator(Box::new(ExactComparator::new("id")))
            .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
            .with_selectivity_hints(vec![0.1, 100.0])
            .set_classifier(Box::new(crate::classify::ThresholdClassifier::new(0.5)));

        let pipeline = builder.build().unwrap();
        // String should be first now (lower sort key)
        assert_eq!(pipeline.comparators[0].field_name(), "name");
        assert_eq!(pipeline.comparators[1].field_name(), "id");
    }

    #[test]
    fn mismatched_hints_length_returns_error() {
        let builder = ReclinkPipeline::builder()
            .add_blocker(Box::new(crate::blocking::ExactBlocking::new("id")))
            .add_comparator(Box::new(ExactComparator::new("id")))
            .add_comparator(Box::new(StringComparator::new("name", Metric::default())))
            .with_selectivity_hints(vec![1.0]) // only 1 hint for 2 comparators
            .set_classifier(Box::new(crate::classify::ThresholdClassifier::new(0.5)));

        let result = builder.build();
        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        };
        assert!(err.contains("selectivity_hints length"));
    }

    #[test]
    fn custom_comparators_get_default_selectivity() {
        use crate::compare::custom::CustomComparatorFn;
        use std::sync::Arc;

        let func: CustomComparatorFn = Arc::new(|a, b| if a == b { 1.0 } else { 0.0 });
        crate::compare::register_custom_comparator("test_sel_default", func).unwrap();

        let cmp = crate::compare::custom_comparator_from_name("field", "test_sel_default").unwrap();
        assert!((cmp.selectivity_hint() - 1.0).abs() < f64::EPSILON);

        crate::compare::unregister_custom_comparator("test_sel_default");
    }
}
