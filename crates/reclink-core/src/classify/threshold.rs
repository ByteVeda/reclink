//! Simple threshold classifier.

use crate::classify::Classifier;
use crate::record::{ClassifiedPair, ComparisonVector, MatchClass};

/// Classifies pairs based on the average comparison score exceeding a threshold.
#[derive(Debug, Clone)]
pub struct ThresholdClassifier {
    /// Score threshold for match classification.
    pub threshold: f64,
}

impl ThresholdClassifier {
    /// Creates a new threshold classifier.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Classifier for ThresholdClassifier {
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair {
        let avg = if vector.scores.is_empty() {
            0.0
        } else {
            vector.scores.iter().sum::<f64>() / vector.scores.len() as f64
        };

        ClassifiedPair {
            pair: vector.pair,
            scores: vector.scores.clone(),
            aggregate_score: avg,
            class: if avg >= self.threshold {
                MatchClass::Match
            } else {
                MatchClass::NonMatch
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::CandidatePair;

    #[test]
    fn above_threshold() {
        let c = ThresholdClassifier::new(0.8);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.9, 0.85],
        };
        assert_eq!(c.classify(&v).class, MatchClass::Match);
    }

    #[test]
    fn below_threshold() {
        let c = ThresholdClassifier::new(0.8);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.5, 0.6],
        };
        assert_eq!(c.classify(&v).class, MatchClass::NonMatch);
    }
}
