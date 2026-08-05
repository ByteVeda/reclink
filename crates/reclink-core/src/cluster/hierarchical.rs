//! Hierarchical agglomerative clustering.

/// Linkage criterion for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Minimum distance between clusters (single linkage).
    Single,
    /// Maximum distance between clusters (complete linkage).
    Complete,
    /// Average distance between clusters.
    Average,
}

/// Hierarchical agglomerative clustering that groups matched pairs
/// using a specified linkage criterion and distance threshold.
#[derive(Debug, Clone)]
pub struct HierarchicalClustering {
    /// The linkage criterion to use.
    pub linkage: Linkage,
    /// Distance threshold: clusters are merged until no pair is closer than this.
    pub threshold: f64,
}

impl HierarchicalClustering {
    /// Creates a new hierarchical clustering instance.
    #[must_use]
    pub fn new(linkage: Linkage, threshold: f64) -> Self {
        Self { linkage, threshold }
    }

    /// Clusters nodes given pairwise similarity scores.
    ///
    /// `similarities` contains `(node_a, node_b, similarity_score)` tuples.
    /// Returns groups of node indices.
    #[must_use]
    pub fn cluster(
        &self,
        num_nodes: usize,
        similarities: &[(usize, usize, f64)],
    ) -> Vec<Vec<usize>> {
        // Convert similarities to distances
        let mut distances: Vec<(usize, usize, f64)> = similarities
            .iter()
            .map(|&(a, b, sim)| (a, b, 1.0 - sim))
            .collect();

        // Initialize each node as its own cluster
        let mut clusters: Vec<Option<Vec<usize>>> = (0..num_nodes).map(|i| Some(vec![i])).collect();

        // Sort by distance (ascending)
        distances.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Track which cluster each node belongs to
        let mut node_cluster: Vec<usize> = (0..num_nodes).collect();

        for (a, b, dist) in distances {
            if dist > self.threshold {
                break;
            }

            let ca = node_cluster[a];
            let cb = node_cluster[b];

            if ca == cb {
                continue;
            }

            // Check linkage criterion
            let should_merge = self.check_linkage(
                clusters[ca].as_ref().unwrap(),
                clusters[cb].as_ref().unwrap(),
                similarities,
            );

            if should_merge {
                let cb_nodes: Vec<usize> = clusters[cb].take().unwrap();
                for &node in &cb_nodes {
                    node_cluster[node] = ca;
                }
                clusters[ca].as_mut().unwrap().extend(cb_nodes);
            }
        }

        clusters
            .into_iter()
            .flatten()
            .filter(|c| c.len() > 1)
            .collect()
    }

    fn check_linkage(
        &self,
        cluster_a: &[usize],
        cluster_b: &[usize],
        similarities: &[(usize, usize, f64)],
    ) -> bool {
        let mut relevant_dists = Vec::new();

        for &(a, b, sim) in similarities {
            let a_in_a = cluster_a.contains(&a);
            let a_in_b = cluster_b.contains(&a);
            let b_in_a = cluster_a.contains(&b);
            let b_in_b = cluster_b.contains(&b);

            if (a_in_a && b_in_b) || (a_in_b && b_in_a) {
                relevant_dists.push(1.0 - sim);
            }
        }

        if relevant_dists.is_empty() {
            return false;
        }

        let criterion_dist = match self.linkage {
            Linkage::Single => relevant_dists.iter().cloned().fold(f64::INFINITY, f64::min),
            Linkage::Complete => relevant_dists
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            Linkage::Average => relevant_dists.iter().sum::<f64>() / relevant_dists.len() as f64,
        };

        criterion_dist <= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_linkage() {
        let hc = HierarchicalClustering::new(Linkage::Single, 0.5);
        let sims = vec![(0, 1, 0.9), (1, 2, 0.7), (3, 4, 0.8)];
        let clusters = hc.cluster(5, &sims);
        assert!(!clusters.is_empty());
    }

    #[test]
    fn complete_linkage_strict() {
        let hc = HierarchicalClustering::new(Linkage::Complete, 0.2);
        let sims = vec![(0, 1, 0.9), (1, 2, 0.6)];
        let clusters = hc.cluster(3, &sims);
        // Complete linkage with threshold 0.2 should only merge very close pairs
        assert!(!clusters.is_empty());
    }
}
