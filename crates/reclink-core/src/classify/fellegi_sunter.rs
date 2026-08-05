//! Fellegi-Sunter probabilistic record linkage classifier.

use crate::classify::Classifier;
use crate::record::{ClassifiedPair, ComparisonVector, MatchClass};

/// Fellegi-Sunter probabilistic classifier using m/u probabilities
/// and log-likelihood ratios.
#[derive(Debug, Clone)]
pub struct FellegiSunterClassifier {
    /// m-probabilities: P(agree | match) for each field.
    pub m_probs: Vec<f64>,
    /// u-probabilities: P(agree | non-match) for each field.
    pub u_probs: Vec<f64>,
    /// Upper threshold for definite matches.
    pub upper_threshold: f64,
    /// Lower threshold below which pairs are definite non-matches.
    pub lower_threshold: f64,
    /// Threshold for agreement on each field (above = agree, below = disagree).
    pub agreement_threshold: f64,
}

impl FellegiSunterClassifier {
    /// Creates a new Fellegi-Sunter classifier.
    #[must_use]
    pub fn new(
        m_probs: Vec<f64>,
        u_probs: Vec<f64>,
        upper_threshold: f64,
        lower_threshold: f64,
    ) -> Self {
        Self {
            m_probs,
            u_probs,
            upper_threshold,
            lower_threshold,
            agreement_threshold: 0.5,
        }
    }
}

impl Classifier for FellegiSunterClassifier {
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair {
        let mut log_likelihood = 0.0;

        for (i, &score) in vector.scores.iter().enumerate() {
            let m = self.m_probs.get(i).copied().unwrap_or(0.9);
            let u = self.u_probs.get(i).copied().unwrap_or(0.1);

            if score >= self.agreement_threshold {
                // Agreement weight
                if u > 0.0 && m > 0.0 {
                    log_likelihood += (m / u).ln();
                }
            } else {
                // Disagreement weight
                let m_disagree = 1.0 - m;
                let u_disagree = 1.0 - u;
                if u_disagree > 0.0 && m_disagree > 0.0 {
                    log_likelihood += (m_disagree / u_disagree).ln();
                }
            }
        }

        let class = if log_likelihood >= self.upper_threshold {
            MatchClass::Match
        } else if log_likelihood <= self.lower_threshold {
            MatchClass::NonMatch
        } else {
            MatchClass::Possible
        };

        ClassifiedPair {
            pair: vector.pair,
            scores: vector.scores.clone(),
            aggregate_score: log_likelihood,
            class,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::CandidatePair;

    #[test]
    fn high_agreement() {
        let c = FellegiSunterClassifier::new(vec![0.95, 0.90], vec![0.05, 0.10], 3.0, -3.0);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.9, 0.85],
        };
        let result = c.classify(&v);
        assert_eq!(result.class, MatchClass::Match);
    }

    #[test]
    fn low_agreement() {
        let c = FellegiSunterClassifier::new(vec![0.95, 0.90], vec![0.05, 0.10], 3.0, -3.0);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.1, 0.1],
        };
        let result = c.classify(&v);
        assert_eq!(result.class, MatchClass::NonMatch);
    }
}
