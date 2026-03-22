//! CART-style decision tree classifier for record linkage.
//!
//! Builds a binary decision tree that splits on comparison vector fields
//! to maximize Gini impurity reduction.

use crate::classify::Classifier;
use crate::record::{ClassifiedPair, ComparisonVector, MatchClass};

/// A node in the decision tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TreeNode {
    /// Leaf node with a prediction.
    Leaf {
        /// Predicted class (true = match).
        prediction: bool,
        /// Match probability (fraction of matches in training data at this leaf).
        probability: f64,
    },
    /// Internal split node.
    Split {
        /// Index of the comparison field to split on.
        feature_index: usize,
        /// Split threshold: left <= threshold, right > threshold.
        threshold: f64,
        /// Left subtree.
        left: Box<TreeNode>,
        /// Right subtree.
        right: Box<TreeNode>,
    },
}

/// Decision tree classifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecisionTreeClassifier {
    /// Root node of the tree.
    pub root: TreeNode,
    /// Probability threshold for match classification.
    pub match_threshold: f64,
}

/// Configuration for decision tree training.
#[derive(Debug, Clone)]
pub struct DecisionTreeConfig {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Minimum samples at a leaf.
    pub min_samples_leaf: usize,
    /// Minimum samples to attempt a split.
    pub min_samples_split: usize,
    /// Probability threshold for match classification.
    pub match_threshold: f64,
}

impl Default for DecisionTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            min_samples_leaf: 5,
            min_samples_split: 10,
            match_threshold: 0.5,
        }
    }
}

/// Trains a decision tree classifier from labeled comparison vectors.
#[must_use]
pub fn train_decision_tree(
    vectors: &[Vec<f64>],
    labels: &[bool],
    config: &DecisionTreeConfig,
) -> DecisionTreeClassifier {
    assert_eq!(vectors.len(), labels.len());
    let indices: Vec<usize> = (0..vectors.len()).collect();
    let num_features = if vectors.is_empty() {
        0
    } else {
        vectors[0].len()
    };
    let root = build_tree(vectors, labels, &indices, num_features, 0, config);
    DecisionTreeClassifier {
        root,
        match_threshold: config.match_threshold,
    }
}

fn build_tree(
    vectors: &[Vec<f64>],
    labels: &[bool],
    indices: &[usize],
    num_features: usize,
    depth: usize,
    config: &DecisionTreeConfig,
) -> TreeNode {
    let n_matches = indices.iter().filter(|&&i| labels[i]).count();
    let n_total = indices.len();
    let prob = if n_total > 0 {
        n_matches as f64 / n_total as f64
    } else {
        0.0
    };

    // Stopping conditions
    if depth >= config.max_depth
        || n_total < config.min_samples_split
        || n_matches == 0
        || n_matches == n_total
    {
        return TreeNode::Leaf {
            prediction: prob >= 0.5,
            probability: prob,
        };
    }

    // Find best split
    let mut best_gain = 0.0f64;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let parent_gini = gini(n_matches, n_total);

    #[allow(clippy::needless_range_loop)]
    for feat in 0..num_features {
        // Collect unique values for this feature (need index, not column)
        let mut values: Vec<f64> = indices.iter().map(|&i| vectors[i][feat]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();

        for window in values.windows(2) {
            let threshold = (window[0] + window[1]) / 2.0;

            let mut left_match = 0;
            let mut left_total = 0;
            let mut right_match = 0;
            let mut right_total = 0;

            for &i in indices {
                if vectors[i][feat] <= threshold {
                    left_total += 1;
                    if labels[i] {
                        left_match += 1;
                    }
                } else {
                    right_total += 1;
                    if labels[i] {
                        right_match += 1;
                    }
                }
            }

            if left_total < config.min_samples_leaf || right_total < config.min_samples_leaf {
                continue;
            }

            let left_gini = gini(left_match, left_total);
            let right_gini = gini(right_match, right_total);
            let weighted =
                (left_total as f64 * left_gini + right_total as f64 * right_gini) / n_total as f64;
            let gain = parent_gini - weighted;

            if gain > best_gain {
                best_gain = gain;
                best_feature = feat;
                best_threshold = threshold;
            }
        }
    }

    if best_gain <= 0.0 {
        return TreeNode::Leaf {
            prediction: prob >= 0.5,
            probability: prob,
        };
    }

    let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
        .iter()
        .partition(|&&i| vectors[i][best_feature] <= best_threshold);

    let left = build_tree(
        vectors,
        labels,
        &left_indices,
        num_features,
        depth + 1,
        config,
    );
    let right = build_tree(
        vectors,
        labels,
        &right_indices,
        num_features,
        depth + 1,
        config,
    );

    TreeNode::Split {
        feature_index: best_feature,
        threshold: best_threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn gini(n_positive: usize, n_total: usize) -> f64 {
    if n_total == 0 {
        return 0.0;
    }
    let p = n_positive as f64 / n_total as f64;
    1.0 - p * p - (1.0 - p) * (1.0 - p)
}

impl DecisionTreeClassifier {
    /// Predicts the match probability for a feature vector.
    #[must_use]
    pub fn predict_probability(&self, scores: &[f64]) -> f64 {
        predict_node(&self.root, scores)
    }
}

fn predict_node(node: &TreeNode, scores: &[f64]) -> f64 {
    match node {
        TreeNode::Leaf { probability, .. } => *probability,
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            if scores.get(*feature_index).copied().unwrap_or(0.0) <= *threshold {
                predict_node(left, scores)
            } else {
                predict_node(right, scores)
            }
        }
    }
}

impl Classifier for DecisionTreeClassifier {
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair {
        let prob = self.predict_probability(&vector.scores);
        ClassifiedPair {
            pair: vector.pair,
            scores: vector.scores.clone(),
            aggregate_score: prob,
            class: if prob >= self.match_threshold {
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
    fn train_separable() {
        let vectors = vec![
            vec![0.9, 0.95],
            vec![0.85, 0.9],
            vec![0.8, 0.85],
            vec![0.1, 0.15],
            vec![0.2, 0.1],
            vec![0.15, 0.2],
        ];
        let labels = vec![true, true, true, false, false, false];
        let config = DecisionTreeConfig {
            max_depth: 3,
            min_samples_leaf: 1,
            min_samples_split: 2,
            ..Default::default()
        };
        let clf = train_decision_tree(&vectors, &labels, &config);

        assert!(clf.predict_probability(&[0.9, 0.9]) > 0.5);
        assert!(clf.predict_probability(&[0.1, 0.1]) < 0.5);
    }

    #[test]
    fn classify_trait() {
        let vectors = vec![vec![0.9], vec![0.1]];
        let labels = vec![true, false];
        let config = DecisionTreeConfig {
            max_depth: 2,
            min_samples_leaf: 1,
            min_samples_split: 1,
            ..Default::default()
        };
        let clf = train_decision_tree(&vectors, &labels, &config);
        let v = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.9],
        };
        assert_eq!(clf.classify(&v).class, MatchClass::Match);
    }

    #[test]
    fn depth_limit() {
        let vectors = vec![vec![0.5], vec![0.5]];
        let labels = vec![true, false];
        let config = DecisionTreeConfig {
            max_depth: 0,
            min_samples_leaf: 1,
            min_samples_split: 1,
            ..Default::default()
        };
        let clf = train_decision_tree(&vectors, &labels, &config);
        // With depth 0, should be a single leaf
        assert!(matches!(clf.root, TreeNode::Leaf { .. }));
    }
}
