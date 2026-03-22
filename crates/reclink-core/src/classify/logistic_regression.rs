//! Logistic regression classifier for record linkage.
//!
//! Learns per-field weights from labeled training data using batch gradient
//! descent with L2 regularization. Implemented from scratch with no external
//! ML dependencies.

use crate::classify::Classifier;
use crate::record::{ClassifiedPair, ComparisonVector, MatchClass};

/// Logistic regression classifier with learned per-field weights.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogisticRegressionClassifier {
    /// Learned weight for each comparison field.
    pub weights: Vec<f64>,
    /// Bias (intercept) term.
    pub bias: f64,
    /// Classification threshold on the predicted probability.
    pub threshold: f64,
}

/// Configuration for logistic regression training.
#[derive(Debug, Clone)]
pub struct LogisticRegressionConfig {
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Stop when max weight change is below this.
    pub convergence_threshold: f64,
    /// L2 regularization strength.
    pub regularization: f64,
    /// Classification threshold.
    pub threshold: f64,
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            regularization: 0.01,
            threshold: 0.5,
        }
    }
}

/// Trains a logistic regression classifier from labeled comparison vectors.
///
/// `vectors` is a slice of feature vectors (one per labeled pair).
/// `labels` is a parallel slice of booleans (true = match).
#[must_use]
pub fn train_logistic_regression(
    vectors: &[Vec<f64>],
    labels: &[bool],
    config: &LogisticRegressionConfig,
) -> LogisticRegressionClassifier {
    assert_eq!(
        vectors.len(),
        labels.len(),
        "vectors and labels must have same length"
    );
    assert!(!vectors.is_empty(), "need at least one training example");

    let num_features = vectors[0].len();
    let n = vectors.len() as f64;
    let mut weights = vec![0.0f64; num_features];
    let mut bias = 0.0f64;

    for _ in 0..config.max_iterations {
        let mut dw = vec![0.0f64; num_features];
        let mut db = 0.0f64;

        for (x, &y) in vectors.iter().zip(labels.iter()) {
            let z: f64 = x
                .iter()
                .zip(weights.iter())
                .map(|(xi, wi)| xi * wi)
                .sum::<f64>()
                + bias;
            let p = sigmoid(z);
            let y_f = if y { 1.0 } else { 0.0 };
            let error = p - y_f;

            for (j, dw_j) in dw.iter_mut().enumerate() {
                *dw_j += error * x[j];
            }
            db += error;
        }

        // Average and add regularization
        let mut max_change = 0.0f64;
        for (j, dw_j) in dw.iter().enumerate() {
            let grad = dw_j / n + config.regularization * weights[j];
            let change = config.learning_rate * grad;
            weights[j] -= change;
            max_change = max_change.max(change.abs());
        }
        let bias_change = config.learning_rate * (db / n);
        bias -= bias_change;
        max_change = max_change.max(bias_change.abs());

        if max_change < config.convergence_threshold {
            break;
        }
    }

    LogisticRegressionClassifier {
        weights,
        bias,
        threshold: config.threshold,
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl LogisticRegressionClassifier {
    /// Predicts the match probability for a comparison vector.
    #[must_use]
    pub fn predict_probability(&self, scores: &[f64]) -> f64 {
        let z: f64 = scores
            .iter()
            .zip(self.weights.iter())
            .map(|(s, w)| s * w)
            .sum::<f64>()
            + self.bias;
        sigmoid(z)
    }
}

impl Classifier for LogisticRegressionClassifier {
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair {
        let prob = self.predict_probability(&vector.scores);
        ClassifiedPair {
            pair: vector.pair,
            scores: vector.scores.clone(),
            aggregate_score: prob,
            class: if prob >= self.threshold {
                MatchClass::Match
            } else {
                MatchClass::NonMatch
            },
        }
    }

    fn can_reject_early(&self, partial_scores: &[f64], total_comparators: usize) -> bool {
        if total_comparators == 0 || self.weights.len() != total_comparators {
            return false;
        }
        // Best case: remaining scores are 1.0
        let mut best_z = self.bias;
        for (i, &w) in self.weights.iter().enumerate() {
            if partial_scores[i] > 0.0 {
                best_z += partial_scores[i] * w;
            } else if w > 0.0 {
                best_z += w; // best case: score = 1.0
            }
            // negative weights with score 1.0 aren't "best case", but we're conservative
        }
        sigmoid(best_z) < self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::CandidatePair;

    #[test]
    fn train_separable_data() {
        // Linearly separable: matches have high scores, non-matches have low
        let vectors = vec![
            vec![0.9, 0.95],
            vec![0.85, 0.9],
            vec![0.1, 0.15],
            vec![0.2, 0.1],
        ];
        let labels = vec![true, true, false, false];
        let config = LogisticRegressionConfig::default();
        let clf = train_logistic_regression(&vectors, &labels, &config);

        assert!(clf.predict_probability(&[0.9, 0.9]) > 0.5);
        assert!(clf.predict_probability(&[0.1, 0.1]) < 0.5);
    }

    #[test]
    fn classify_trait() {
        let clf = LogisticRegressionClassifier {
            weights: vec![2.0, 2.0],
            bias: -2.0,
            threshold: 0.5,
        };
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.9, 0.9],
        };
        assert_eq!(clf.classify(&v).class, MatchClass::Match);

        let v2 = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.1, 0.1],
        };
        assert_eq!(clf.classify(&v2).class, MatchClass::NonMatch);
    }

    #[test]
    fn convergence() {
        let vectors = vec![vec![1.0], vec![0.0]];
        let labels = vec![true, false];
        let config = LogisticRegressionConfig {
            max_iterations: 10000,
            ..Default::default()
        };
        let clf = train_logistic_regression(&vectors, &labels, &config);
        assert!(clf.weights[0] > 0.0, "Weight should be positive");
    }
}
