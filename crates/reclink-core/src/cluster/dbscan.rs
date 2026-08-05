//! DBSCAN density-based clustering.
//!
//! Discovers clusters by density without requiring a predefined cluster count.
//! Works with pairwise similarity data (consistent with the record linkage pipeline).

use ahash::AHashMap;

/// Result of DBSCAN clustering.
#[derive(Debug, Clone)]
pub struct DbscanResult {
    /// Groups of record indices (one per cluster).
    pub clusters: Vec<Vec<usize>>,
    /// Indices of noise points (not assigned to any cluster).
    pub noise: Vec<usize>,
    /// Cluster label for each node (-1 for noise).
    pub labels: Vec<i32>,
}

/// DBSCAN density-based clustering.
///
/// Two points are neighbors if their similarity is >= `min_similarity`.
/// A core point has at least `min_samples` neighbors. Clusters are formed
/// by expanding from core points through density-reachable chains.
pub struct Dbscan {
    /// Minimum similarity for two points to be neighbors.
    pub min_similarity: f64,
    /// Minimum number of neighbors for a core point.
    pub min_samples: usize,
}

impl Dbscan {
    /// Creates a new DBSCAN clusterer.
    #[must_use]
    pub fn new(min_similarity: f64, min_samples: usize) -> Self {
        Self {
            min_similarity,
            min_samples,
        }
    }

    /// Clusters nodes given pairwise similarity scores.
    #[must_use]
    pub fn cluster(&self, num_nodes: usize, similarities: &[(usize, usize, f64)]) -> DbscanResult {
        let adj = build_adjacency(num_nodes, similarities, self.min_similarity);
        let mut labels = vec![-1i32; num_nodes];
        let mut cluster_id = 0i32;

        for node in 0..num_nodes {
            if labels[node] != -1 {
                continue;
            }
            let neighbors = &adj[node];
            if neighbors.len() < self.min_samples {
                continue; // not a core point, stays noise for now
            }

            // Expand cluster
            labels[node] = cluster_id;
            let mut queue: Vec<usize> = neighbors.clone();
            let mut qi = 0;

            while qi < queue.len() {
                let q = queue[qi];
                qi += 1;

                if labels[q] == -1 {
                    labels[q] = cluster_id; // was noise, now border
                } else if labels[q] != -1 && labels[q] != cluster_id {
                    continue; // already assigned
                } else if labels[q] == cluster_id {
                    // already in this cluster
                }

                if labels[q] != cluster_id {
                    labels[q] = cluster_id;
                }

                let q_neighbors = &adj[q];
                if q_neighbors.len() >= self.min_samples {
                    // q is also a core point, expand
                    for &nn in q_neighbors {
                        if labels[nn] == -1 {
                            queue.push(nn);
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        // Build result
        let mut clusters: AHashMap<i32, Vec<usize>> = AHashMap::new();
        let mut noise = Vec::new();

        for (i, &label) in labels.iter().enumerate() {
            if label == -1 {
                noise.push(i);
            } else {
                clusters.entry(label).or_default().push(i);
            }
        }

        let mut cluster_list: Vec<(i32, Vec<usize>)> = clusters.into_iter().collect();
        cluster_list.sort_by_key(|(id, _)| *id);
        let clusters = cluster_list.into_iter().map(|(_, v)| v).collect();

        DbscanResult {
            clusters,
            noise,
            labels,
        }
    }
}

/// Builds adjacency lists from similarity triples.
pub(crate) fn build_adjacency(
    num_nodes: usize,
    similarities: &[(usize, usize, f64)],
    min_similarity: f64,
) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); num_nodes];
    for &(a, b, sim) in similarities {
        if sim >= min_similarity && a < num_nodes && b < num_nodes {
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters() {
        // Nodes 0,1,2 form one cluster, nodes 3,4,5 form another
        let sims = vec![
            (0, 1, 0.9),
            (1, 2, 0.9),
            (0, 2, 0.85),
            (3, 4, 0.9),
            (4, 5, 0.9),
            (3, 5, 0.85),
        ];
        let db = Dbscan::new(0.8, 2);
        let result = db.cluster(6, &sims);
        assert_eq!(result.clusters.len(), 2);
        assert!(result.noise.is_empty());
    }

    #[test]
    fn noise_points() {
        let sims = vec![(0, 1, 0.9), (1, 2, 0.9)];
        let db = Dbscan::new(0.8, 2);
        let result = db.cluster(4, &sims); // node 3 has no edges
        assert!(!result.noise.is_empty());
    }

    #[test]
    fn empty() {
        let db = Dbscan::new(0.8, 2);
        let result = db.cluster(0, &[]);
        assert!(result.clusters.is_empty());
        assert!(result.noise.is_empty());
    }

    #[test]
    fn all_noise() {
        // No edges meet min_similarity
        let sims = vec![(0, 1, 0.5)];
        let db = Dbscan::new(0.8, 2);
        let result = db.cluster(3, &sims);
        assert!(result.clusters.is_empty());
        assert_eq!(result.noise.len(), 3);
    }

    #[test]
    fn labels_consistent() {
        let sims = vec![(0, 1, 0.9), (1, 2, 0.9), (0, 2, 0.85)];
        let db = Dbscan::new(0.8, 2);
        let result = db.cluster(3, &sims);
        for cluster in &result.clusters {
            let label = result.labels[cluster[0]];
            for &idx in cluster {
                assert_eq!(result.labels[idx], label);
            }
        }
    }
}
