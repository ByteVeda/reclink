//! BK-tree for efficient metric-space nearest-neighbor search.
//!
//! Uses the triangle inequality to prune branches:
//! if `distance(query, node) = d` and we want `distance <= k`,
//! only explore children with edge labels in `[d-k, d+k]`.

use ahash::AHashMap;

use crate::error::{ReclinkError, Result};
use crate::metrics::{DistanceMetric, Metric};

/// Result from a BK-tree search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BkSearchResult {
    /// The matched string value.
    pub value: String,
    /// Index of the string in the original build list.
    pub index: usize,
    /// Distance from the query.
    pub distance: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BkNode {
    value: String,
    index: usize,
    children: AHashMap<usize, BkNode>,
}

/// A BK-tree for efficient metric-space nearest-neighbor search.
///
/// Only works with integer distance metrics (Levenshtein, Damerau-Levenshtein, Hamming).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BkTree {
    root: Option<Box<BkNode>>,
    metric: Metric,
    size: usize,
}

fn compute_distance_with(metric: &Metric, a: &str, b: &str) -> usize {
    match metric {
        Metric::Levenshtein(m) => m.distance(a, b).unwrap_or(0),
        Metric::DamerauLevenshtein(m) => m.distance(a, b).unwrap_or(0),
        Metric::Hamming(m) => m.distance(a, b).unwrap_or(0),
        _ => 0,
    }
}

impl BkTree {
    /// Build a new BK-tree from a list of strings.
    ///
    /// # Errors
    ///
    /// Returns an error if the metric is not an integer distance metric.
    pub fn build(strings: &[&str], metric: Metric) -> Result<Self> {
        Self::validate_metric(&metric)?;
        let mut tree = Self {
            root: None,
            metric,
            size: 0,
        };
        for (i, s) in strings.iter().enumerate() {
            tree.insert(s, i);
        }
        Ok(tree)
    }

    /// Insert a single string into the tree.
    pub fn insert(&mut self, s: &str, index: usize) {
        self.size += 1;
        let new_node = BkNode {
            value: s.to_string(),
            index,
            children: AHashMap::new(),
        };

        if self.root.is_none() {
            self.root = Some(Box::new(new_node));
            return;
        }

        let metric = &self.metric;
        let mut current = self.root.as_deref_mut().unwrap();
        loop {
            let d = compute_distance_with(metric, &current.value, s);
            if current.children.contains_key(&d) {
                current = current.children.get_mut(&d).unwrap();
            } else {
                current.children.insert(d, new_node);
                return;
            }
        }
    }

    /// Find all strings within `max_distance` of the query.
    #[must_use]
    pub fn find_within(&self, query: &str, max_distance: usize) -> Vec<BkSearchResult> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            self.search_node(root, query, max_distance, &mut results);
        }
        results.sort_by_key(|r| r.distance);
        results
    }

    /// Find the k nearest neighbors of the query.
    #[must_use]
    pub fn find_nearest(&self, query: &str, k: usize) -> Vec<BkSearchResult> {
        if self.root.is_none() || k == 0 {
            return Vec::new();
        }

        // Start with a large radius and narrow down
        let mut best_distance = usize::MAX;
        let mut results = Vec::new();
        self.knn_search(
            self.root.as_ref().unwrap(),
            query,
            k,
            &mut results,
            &mut best_distance,
        );

        results.sort_by_key(|r| r.distance);
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

    fn validate_metric(metric: &Metric) -> Result<()> {
        match metric {
            Metric::Levenshtein(_) | Metric::DamerauLevenshtein(_) | Metric::Hamming(_) => Ok(()),
            _ => Err(ReclinkError::InvalidConfig(
                "BK-tree requires an integer distance metric (levenshtein, \
                 damerau_levenshtein, or hamming)"
                    .to_string(),
            )),
        }
    }

    fn search_node(
        &self,
        node: &BkNode,
        query: &str,
        max_distance: usize,
        results: &mut Vec<BkSearchResult>,
    ) {
        let d = compute_distance_with(&self.metric, &node.value, query);
        if d <= max_distance {
            results.push(BkSearchResult {
                value: node.value.clone(),
                index: node.index,
                distance: d,
            });
        }

        let low = d.saturating_sub(max_distance);
        let high = d + max_distance;
        for (&edge_dist, child) in &node.children {
            if edge_dist >= low && edge_dist <= high {
                self.search_node(child, query, max_distance, results);
            }
        }
    }

    fn knn_search(
        &self,
        node: &BkNode,
        query: &str,
        k: usize,
        results: &mut Vec<BkSearchResult>,
        best_distance: &mut usize,
    ) {
        let d = compute_distance_with(&self.metric, &node.value, query);

        if results.len() < k {
            results.push(BkSearchResult {
                value: node.value.clone(),
                index: node.index,
                distance: d,
            });
            results.sort_by_key(|r| r.distance);
            if results.len() == k {
                *best_distance = results[k - 1].distance;
            }
        } else if d < *best_distance {
            results.push(BkSearchResult {
                value: node.value.clone(),
                index: node.index,
                distance: d,
            });
            results.sort_by_key(|r| r.distance);
            results.truncate(k);
            *best_distance = results[k - 1].distance;
        }

        let low = d.saturating_sub(*best_distance);
        let high = d.saturating_add(*best_distance);
        for (&edge_dist, child) in &node.children {
            if edge_dist >= low && edge_dist <= high {
                self.knn_search(child, query, k, results, best_distance);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Levenshtein;

    #[test]
    fn build_and_query() {
        let tree = BkTree::build(
            &["hello", "hallo", "world", "help"],
            Metric::Levenshtein(Levenshtein),
        )
        .unwrap();
        let results = tree.find_within("hello", 1);
        let values: Vec<&str> = results.iter().map(|r| r.value.as_str()).collect();
        assert!(values.contains(&"hello"));
        assert!(values.contains(&"hallo"));
        assert!(!values.contains(&"world"));
    }

    #[test]
    fn find_nearest() {
        let tree = BkTree::build(
            &["apple", "apply", "ape", "banana"],
            Metric::Levenshtein(Levenshtein),
        )
        .unwrap();
        let results = tree.find_nearest("appel", 2);
        assert_eq!(results.len(), 2);
        // "apple" should be closest (distance 1)
        assert_eq!(results[0].value, "apple");
    }

    #[test]
    fn empty_tree() {
        let tree: BkTree = BkTree::build(&[], Metric::Levenshtein(Levenshtein)).unwrap();
        assert!(tree.is_empty());
        assert_eq!(tree.find_within("hello", 1).len(), 0);
        assert_eq!(tree.find_nearest("hello", 1).len(), 0);
    }

    #[test]
    fn invalid_metric() {
        let result = BkTree::build(&["hello"], Metric::Jaro(crate::metrics::Jaro));
        assert!(result.is_err());
    }

    #[test]
    fn len() {
        let tree = BkTree::build(&["a", "b", "c"], Metric::Levenshtein(Levenshtein)).unwrap();
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn exact_match() {
        let tree = BkTree::build(&["hello", "world"], Metric::Levenshtein(Levenshtein)).unwrap();
        let results = tree.find_within("hello", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, "hello");
        assert_eq!(results[0].distance, 0);
    }
}
