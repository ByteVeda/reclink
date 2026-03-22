//! Classifiers for turning comparison vectors into match/non-match decisions.

pub mod custom;
mod decision_tree;
mod em;
mod fellegi_sunter;
mod logistic_regression;
mod threshold;
mod weighted;

pub use custom::*;
pub use decision_tree::{
    train_decision_tree, DecisionTreeClassifier, DecisionTreeConfig, TreeNode,
};
pub use em::*;
pub use fellegi_sunter::FellegiSunterClassifier;
pub use logistic_regression::{
    train_logistic_regression, LogisticRegressionClassifier, LogisticRegressionConfig,
};
pub use threshold::{ThresholdBandsClassifier, ThresholdClassifier};
pub use weighted::{WeightedSumBandsClassifier, WeightedSumClassifier};

use crate::record::{ClassifiedPair, ComparisonVector};

/// Trait for classifiers that decide whether a comparison vector represents a match.
pub trait Classifier: Send + Sync {
    /// Classifies a comparison vector into a match decision.
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair;

    /// Returns `true` if the classifier can definitively reject this pair
    /// based on partial scores (remaining scores are assumed to be 0.0).
    ///
    /// `partial_scores` has the same length as `total_comparators`; already-computed
    /// positions contain real scores, positions not yet evaluated contain 0.0.
    fn can_reject_early(&self, _partial_scores: &[f64], _total_comparators: usize) -> bool {
        false
    }
}
