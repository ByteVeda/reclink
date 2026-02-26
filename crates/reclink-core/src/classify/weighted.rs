//! Weighted sum classifier.

use crate::classify::Classifier;
use crate::record::{ClassifiedPair, ComparisonVector, MatchClass};

/// Classifies pairs using a weighted sum of comparison scores.
#[derive(Debug, Clone)]
pub struct WeightedSumClassifier {
    /// Per-field weights (must match the number of comparators).
    pub weights: Vec<f64>,
    /// Score threshold for match classification.
    pub threshold: f64,
}

impl WeightedSumClassifier {
    /// Creates a new weighted sum classifier.
    #[must_use]
    pub fn new(weights: Vec<f64>, threshold: f64) -> Self {
        Self { weights, threshold }
    }
}

impl Classifier for WeightedSumClassifier {
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair {
        let weighted_sum: f64 = vector
            .scores
            .iter()
            .zip(self.weights.iter())
            .map(|(s, w)| s * w)
            .sum();

        ClassifiedPair {
            pair: vector.pair,
            scores: vector.scores.clone(),
            aggregate_score: weighted_sum,
            class: if weighted_sum >= self.threshold {
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
    fn weighted_match() {
        let c = WeightedSumClassifier::new(vec![0.5, 0.5], 0.8);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.9, 0.9],
        };
        let result = c.classify(&v);
        assert_eq!(result.class, MatchClass::Match);
        assert!((result.aggregate_score - 0.9).abs() < 1e-10);
    }
}
