//! BK-tree for efficient metric-space nearest-neighbor search.
//!
//! Uses the triangle inequality to prune branches:
//! if `distance(query, node) = d` and we want `distance <= k`,
//! only explore children with edge labels in `[d-k, d+k]`.

use ahash::{AHashMap, AHashSet};

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
    #[serde(default)]
    deleted: AHashSet<usize>,
    #[serde(default)]
    next_index: usize,
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
            deleted: AHashSet::new(),
            next_index: strings.len(),
        };
        for (i, s) in strings.iter().enumerate() {
            tree.insert(s, i);
        }
        Ok(tree)
    }

    /// Insert a single string into the tree with a specific index.
    fn insert(&mut self, s: &str, index: usize) {
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

    /// Insert a new string and return its assigned index.
    pub fn insert_new(&mut self, s: &str) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        self.insert(s, index);
        index
    }

    /// Soft-delete a string by index. Returns `true` if the index was valid
    /// and not already deleted.
    pub fn remove(&mut self, index: usize) -> bool {
        if index >= self.next_index || self.deleted.contains(&index) {
            return false;
        }
        self.deleted.insert(index);
        self.size = self.size.saturating_sub(1);
        true
    }

    /// Returns `true` if the index is valid and not deleted.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        index < self.next_index && !self.deleted.contains(&index)
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

    /// Estimates the heap memory usage of this tree in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        if let Some(root) = &self.root {
            bytes += Self::node_memory_usage(root);
        }
        // deleted set overhead
        bytes += self.deleted.capacity() * std::mem::size_of::<usize>();
        bytes
    }

    fn node_memory_usage(node: &BkNode) -> usize {
        let mut bytes = std::mem::size_of::<BkNode>();
        bytes += node.value.capacity();
        // AHashMap overhead: each entry is (key, value) + control bytes
        bytes += node.children.capacity()
            * (std::mem::size_of::<usize>() + std::mem::size_of::<BkNode>());
        for (_, child) in &node.children {
            bytes += Self::node_memory_usage(child);
        }
        bytes
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
        if d <= max_distance && !self.deleted.contains(&node.index) {
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

        if !self.deleted.contains(&node.index) {
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

    #[test]
    fn insert_after_build() {
        let mut tree =
            BkTree::build(&["hello", "world"], Metric::Levenshtein(Levenshtein)).unwrap();
        let idx = tree.insert_new("hallo");
        assert_eq!(idx, 2);
        assert_eq!(tree.len(), 3);
        let results = tree.find_within("hallo", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 2);
    }

    #[test]
    fn remove_excludes_from_search() {
        let mut tree = BkTree::build(
            &["hello", "hallo", "world"],
            Metric::Levenshtein(Levenshtein),
        )
        .unwrap();
        assert!(tree.contains(1));
        assert!(tree.remove(1)); // remove "hallo"
        assert!(!tree.contains(1));
        assert_eq!(tree.len(), 2);

        let results = tree.find_within("hallo", 0);
        assert!(results.is_empty());

        // knn should also skip deleted
        let knn = tree.find_nearest("hallo", 1);
        assert_eq!(knn.len(), 1);
        assert_ne!(knn[0].index, 1);
    }

    #[test]
    fn remove_idempotent() {
        let mut tree =
            BkTree::build(&["hello", "world"], Metric::Levenshtein(Levenshtein)).unwrap();
        assert!(tree.remove(0));
        assert!(!tree.remove(0)); // already deleted
        assert!(!tree.remove(99)); // invalid index
    }
}
