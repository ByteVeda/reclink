//! Vantage-point tree for efficient nearest-neighbor search with floating-point metrics.
//!
//! Unlike BK-tree (integer distances only), VP-tree works with any metric
//! that returns a dissimilarity score (1 - similarity).

use serde::{Deserialize, Serialize};

use crate::metrics::Metric;

/// Result from a VP-tree search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpSearchResult {
    /// The matched string value.
    pub value: String,
    /// Index of the string in the original build list.
    pub index: usize,
    /// Distance (1 - similarity) from the query.
    pub distance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VpNode {
    value: String,
    index: usize,
    mu: f64,
    left: Option<Box<VpNode>>,
    right: Option<Box<VpNode>>,
}

/// A vantage-point tree for efficient nearest-neighbor search with any metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpTree {
    root: Option<Box<VpNode>>,
    metric: Metric,
    size: usize,
}

impl VpTree {
    /// Build a VP-tree from a list of strings.
    #[must_use]
    pub fn build(strings: &[&str], metric: Metric) -> Self {
        let mut items: Vec<(String, usize)> = strings
            .iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i))
            .collect();
        let root = Self::build_node(&mut items, &metric);
        Self {
            root,
            metric,
            size: strings.len(),
        }
    }

    /// Find all strings within `max_distance` (dissimilarity) of the query.
    #[must_use]
    pub fn find_within(&self, query: &str, max_distance: f64) -> Vec<VpSearchResult> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            self.search_within(root, query, max_distance, &mut results);
        }
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }

    /// Find the k nearest neighbors of the query.
    #[must_use]
    pub fn find_nearest(&self, query: &str, k: usize) -> Vec<VpSearchResult> {
        if self.root.is_none() || k == 0 {
            return Vec::new();
        }
        let mut results = Vec::new();
        let mut tau = f64::INFINITY;
        self.knn_search(
            self.root.as_ref().unwrap(),
            query,
            k,
            &mut results,
            &mut tau,
        );
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        results
    }

    /// Returns the number of strings in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns whether the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn dissimilarity(metric: &Metric, a: &str, b: &str) -> f64 {
        1.0 - metric.similarity(a, b)
    }

    fn build_node(items: &mut [(String, usize)], metric: &Metric) -> Option<Box<VpNode>> {
        if items.is_empty() {
            return None;
        }
        if items.len() == 1 {
            return Some(Box::new(VpNode {
                value: items[0].0.clone(),
                index: items[0].1,
                mu: 0.0,
                left: None,
                right: None,
            }));
        }

        // Use last element as vantage point (avoids need for random)
        let vp_idx = items.len() - 1;
        let vp_value = items[vp_idx].0.clone();
        let vp_index = items[vp_idx].1;

        // Compute distances from vantage point to all others
        let mut distances: Vec<(f64, usize)> = items[..vp_idx]
            .iter()
            .enumerate()
            .map(|(i, (s, _))| (Self::dissimilarity(metric, &vp_value, s), i))
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let median_idx = distances.len() / 2;
        let mu = if distances.is_empty() {
            0.0
        } else {
            distances[median_idx].0
        };

        // Partition items into left (d <= mu) and right (d > mu)
        let mut left_items: Vec<(String, usize)> = Vec::new();
        let mut right_items: Vec<(String, usize)> = Vec::new();
        for &(d, idx) in &distances {
            let item = items[idx].clone();
            if d <= mu {
                left_items.push(item);
            } else {
                right_items.push(item);
            }
        }

        let left = Self::build_node(&mut left_items, metric);
        let right = Self::build_node(&mut right_items, metric);

        Some(Box::new(VpNode {
            value: vp_value,
            index: vp_index,
            mu,
            left,
            right,
        }))
    }

    fn search_within(
        &self,
        node: &VpNode,
        query: &str,
        max_distance: f64,
        results: &mut Vec<VpSearchResult>,
    ) {
        let d = Self::dissimilarity(&self.metric, &node.value, query);
        if d <= max_distance {
            results.push(VpSearchResult {
                value: node.value.clone(),
                index: node.index,
                distance: d,
            });
        }

        // Search left subtree if d - max_distance <= mu
        if d <= node.mu + max_distance {
            if let Some(left) = &node.left {
                self.search_within(left, query, max_distance, results);
            }
        }
        // Search right subtree if d + max_distance > mu
        if d + max_distance > node.mu {
            if let Some(right) = &node.right {
                self.search_within(right, query, max_distance, results);
            }
        }
    }

    fn knn_search(
        &self,
        node: &VpNode,
        query: &str,
        k: usize,
        results: &mut Vec<VpSearchResult>,
        tau: &mut f64,
    ) {
        let d = Self::dissimilarity(&self.metric, &node.value, query);

        if results.len() < k {
            results.push(VpSearchResult {
                value: node.value.clone(),
                index: node.index,
                distance: d,
            });
            results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            if results.len() == k {
                *tau = results[k - 1].distance;
            }
        } else if d < *tau {
            results.push(VpSearchResult {
                value: node.value.clone(),
                index: node.index,
                distance: d,
            });
            results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            results.truncate(k);
            *tau = results[k - 1].distance;
        }

        // Search children
        if d <= node.mu + *tau {
            if let Some(left) = &node.left {
                self.knn_search(left, query, k, results, tau);
            }
        }
        if d + *tau > node.mu {
            if let Some(right) = &node.right {
                self.knn_search(right, query, k, results, tau);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::JaroWinkler;

    #[test]
    fn build_and_find_within() {
        let tree = VpTree::build(
            &["hello", "hallo", "world", "help"],
            Metric::JaroWinkler(JaroWinkler::default()),
        );
        // "hello" vs "hello" = distance 0
        let results = tree.find_within("hello", 0.2);
        let values: Vec<&str> = results.iter().map(|r| r.value.as_str()).collect();
        assert!(values.contains(&"hello"));
        assert!(values.contains(&"hallo"));
    }

    #[test]
    fn find_nearest() {
        let tree = VpTree::build(
            &["apple", "apply", "ape", "banana"],
            Metric::JaroWinkler(JaroWinkler::default()),
        );
        let results = tree.find_nearest("apple", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].value, "apple");
        assert!((results[0].distance - 0.0).abs() < 1e-10);
    }

    #[test]
    fn empty_tree() {
        let tree = VpTree::build(&[], Metric::JaroWinkler(JaroWinkler::default()));
        assert!(tree.is_empty());
        assert!(tree.find_within("hello", 0.5).is_empty());
        assert!(tree.find_nearest("hello", 1).is_empty());
    }

    #[test]
    fn len() {
        let tree = VpTree::build(
            &["a", "b", "c"],
            Metric::JaroWinkler(JaroWinkler::default()),
        );
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn exact_match() {
        let tree = VpTree::build(
            &["hello", "world"],
            Metric::JaroWinkler(JaroWinkler::default()),
        );
        let results = tree.find_within("hello", 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, "hello");
    }
}
