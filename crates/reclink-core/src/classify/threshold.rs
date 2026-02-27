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

/// Classifies pairs into three bands using average score:
/// `Match` (>= upper), `Possible` (between lower and upper), `NonMatch` (< lower).
#[derive(Debug, Clone)]
pub struct ThresholdBandsClassifier {
    /// Upper threshold: scores at or above this are definite matches.
    pub upper: f64,
    /// Lower threshold: scores below this are definite non-matches.
    pub lower: f64,
}

impl ThresholdBandsClassifier {
    /// Creates a new three-band threshold classifier.
    #[must_use]
    pub fn new(upper: f64, lower: f64) -> Self {
        Self { upper, lower }
    }
}

impl Classifier for ThresholdBandsClassifier {
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair {
        let avg = if vector.scores.is_empty() {
            0.0
        } else {
            vector.scores.iter().sum::<f64>() / vector.scores.len() as f64
        };

        let class = if avg >= self.upper {
            MatchClass::Match
        } else if avg < self.lower {
            MatchClass::NonMatch
        } else {
            MatchClass::Possible
        };

        ClassifiedPair {
            pair: vector.pair,
            scores: vector.scores.clone(),
            aggregate_score: avg,
            class,
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

    #[test]
    fn bands_match() {
        let c = ThresholdBandsClassifier::new(0.8, 0.5);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.9, 0.85],
        };
        assert_eq!(c.classify(&v).class, MatchClass::Match);
    }

    #[test]
    fn bands_possible() {
        let c = ThresholdBandsClassifier::new(0.8, 0.5);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.6, 0.7],
        };
        assert_eq!(c.classify(&v).class, MatchClass::Possible);
    }

    #[test]
    fn bands_non_match() {
        let c = ThresholdBandsClassifier::new(0.8, 0.5);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.3, 0.4],
        };
        assert_eq!(c.classify(&v).class, MatchClass::NonMatch);
    }
}
