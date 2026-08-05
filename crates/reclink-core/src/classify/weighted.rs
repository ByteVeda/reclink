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

    fn can_reject_early(&self, partial_scores: &[f64], _total_comparators: usize) -> bool {
        // Current weighted sum + max possible from remaining (weight * 1.0)
        let current_sum: f64 = partial_scores
            .iter()
            .zip(self.weights.iter())
            .map(|(s, w)| s * w)
            .sum();
        let remaining_max: f64 = partial_scores
            .iter()
            .zip(self.weights.iter())
            .filter(|(&s, _)| s == 0.0)
            .map(|(_, w)| *w)
            .sum();
        current_sum + remaining_max < self.threshold
    }
}

/// Classifies pairs into three bands using weighted sum:
/// `Match` (>= upper), `Possible` (between lower and upper), `NonMatch` (< lower).
#[derive(Debug, Clone)]
pub struct WeightedSumBandsClassifier {
    /// Per-field weights (must match the number of comparators).
    pub weights: Vec<f64>,
    /// Upper threshold: weighted sum at or above this is a definite match.
    pub upper: f64,
    /// Lower threshold: weighted sum below this is a definite non-match.
    pub lower: f64,
}

impl WeightedSumBandsClassifier {
    /// Creates a new three-band weighted sum classifier.
    #[must_use]
    pub fn new(weights: Vec<f64>, upper: f64, lower: f64) -> Self {
        Self {
            weights,
            upper,
            lower,
        }
    }
}

impl Classifier for WeightedSumBandsClassifier {
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair {
        let weighted_sum: f64 = vector
            .scores
            .iter()
            .zip(self.weights.iter())
            .map(|(s, w)| s * w)
            .sum();

        let class = if weighted_sum >= self.upper {
            MatchClass::Match
        } else if weighted_sum < self.lower {
            MatchClass::NonMatch
        } else {
            MatchClass::Possible
        };

        ClassifiedPair {
            pair: vector.pair,
            scores: vector.scores.clone(),
            aggregate_score: weighted_sum,
            class,
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

    #[test]
    fn weighted_bands_match() {
        let c = WeightedSumBandsClassifier::new(vec![0.5, 0.5], 0.8, 0.5);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.9, 0.9],
        };
        let result = c.classify(&v);
        assert_eq!(result.class, MatchClass::Match);
    }

    #[test]
    fn weighted_bands_possible() {
        let c = WeightedSumBandsClassifier::new(vec![0.5, 0.5], 0.8, 0.5);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.7, 0.7],
        };
        let result = c.classify(&v);
        assert_eq!(result.class, MatchClass::Possible);
    }

    #[test]
    fn weighted_bands_non_match() {
        let c = WeightedSumBandsClassifier::new(vec![0.5, 0.5], 0.8, 0.5);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.3, 0.4],
        };
        let result = c.classify(&v);
        assert_eq!(result.class, MatchClass::NonMatch);
    }
}
