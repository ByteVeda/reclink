//! Classifiers for turning comparison vectors into match/non-match decisions.

mod em;
mod fellegi_sunter;
mod threshold;
mod weighted;

pub use em::*;
pub use fellegi_sunter::FellegiSunterClassifier;
pub use threshold::ThresholdClassifier;
pub use weighted::WeightedSumClassifier;

use crate::record::{ClassifiedPair, ComparisonVector};

/// Trait for classifiers that decide whether a comparison vector represents a match.
pub trait Classifier: Send + Sync {
    /// Classifies a comparison vector into a match decision.
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair;
}
