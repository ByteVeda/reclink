//! OPTICS (Ordering Points To Identify Clustering Structure).
//!
//! Extension of DBSCAN for varying-density clusters. Produces a reachability
//! ordering from which clusters can be extracted at various density levels.

use std::collections::BinaryHeap;

use ahash::AHashMap;

// OPTICS builds its own neighbor lists with distances, so it doesn't use
// the shared build_adjacency helper from DBSCAN.

/// A point in the OPTICS ordering with its reachability and core distances.
#[derive(Debug, Clone)]
pub struct OpticsPoint {
    /// Record index.
    pub index: usize,
    /// Reachability distance (1 - similarity). None for first point.
    pub reachability: Option<f64>,
    /// Core distance. None if not a core point.
    pub core_distance: Option<f64>,
}

/// Result of OPTICS clustering.
#[derive(Debug, Clone)]
pub struct OpticsResult {
    /// Points in reachability ordering.
    pub ordering: Vec<OpticsPoint>,
    /// Clusters extracted from the ordering.
    pub clusters: Vec<Vec<usize>>,
    /// Noise point indices.
    pub noise: Vec<usize>,
}

/// OPTICS clustering algorithm.
pub struct Optics {
    /// Minimum neighbors for a core point.
    pub min_samples: usize,
    /// Similarity threshold for extracting flat clusters.
    pub extract_threshold: f64,
}

impl Optics {
    /// Creates a new OPTICS clusterer.
    #[must_use]
    pub fn new(min_samples: usize, extract_threshold: f64) -> Self {
        Self {
            min_samples,
            extract_threshold,
        }
    }

    /// Runs OPTICS and extracts flat clusters at `extract_threshold`.
    #[must_use]
    pub fn cluster(&self, num_nodes: usize, similarities: &[(usize, usize, f64)]) -> OpticsResult {
        let ordering = self.compute_ordering(num_nodes, similarities);
        let (clusters, noise) = self.extract_clusters(&ordering, num_nodes);
        OpticsResult {
            ordering,
            clusters,
            noise,
        }
    }

    /// Computes the OPTICS reachability ordering.
    fn compute_ordering(
        &self,
        num_nodes: usize,
        similarities: &[(usize, usize, f64)],
    ) -> Vec<OpticsPoint> {
        // Build neighbor map with distances (1 - similarity)
        let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_nodes];
        for &(a, b, sim) in similarities {
            if a < num_nodes && b < num_nodes {
                let dist = 1.0 - sim;
                neighbors[a].push((b, dist));
                neighbors[b].push((a, dist));
            }
        }

        // Sort neighbors by distance for core distance computation
        for n in &mut neighbors {
            n.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Compute core distances
        let core_distances: Vec<Option<f64>> = (0..num_nodes)
            .map(|i| {
                if neighbors[i].len() >= self.min_samples {
                    Some(neighbors[i][self.min_samples - 1].1)
                } else {
                    None
                }
            })
            .collect();

        let mut processed = vec![false; num_nodes];
        let mut ordering = Vec::with_capacity(num_nodes);
        let mut reachability = vec![f64::INFINITY; num_nodes];

        for start in 0..num_nodes {
            if processed[start] {
                continue;
            }

            processed[start] = true;
            ordering.push(OpticsPoint {
                index: start,
                reachability: None,
                core_distance: core_distances[start],
            });

            if core_distances[start].is_none() {
                continue;
            }

            // Seed the priority queue
            let mut heap = BinaryHeap::new();
            self.update_seeds(
                start,
                &neighbors,
                &core_distances,
                &processed,
                &mut reachability,
                &mut heap,
            );

            while let Some(OrderedPoint { index: current, .. }) = heap.pop() {
                if processed[current] {
                    continue;
                }
                processed[current] = true;
                ordering.push(OpticsPoint {
                    index: current,
                    reachability: Some(reachability[current]),
                    core_distance: core_distances[current],
                });

                if core_distances[current].is_some() {
                    self.update_seeds(
                        current,
                        &neighbors,
                        &core_distances,
                        &processed,
                        &mut reachability,
                        &mut heap,
                    );
                }
            }
        }

        ordering
    }

    fn update_seeds(
        &self,
        point: usize,
        neighbors: &[Vec<(usize, f64)>],
        core_distances: &[Option<f64>],
        processed: &[bool],
        reachability: &mut [f64],
        heap: &mut BinaryHeap<OrderedPoint>,
    ) {
        let cd = match core_distances[point] {
            Some(cd) => cd,
            None => return,
        };

        for &(neighbor, dist) in &neighbors[point] {
            if processed[neighbor] {
                continue;
            }
            let new_reach = cd.max(dist);
            if new_reach < reachability[neighbor] {
                reachability[neighbor] = new_reach;
                heap.push(OrderedPoint {
                    index: neighbor,
                    distance: new_reach,
                });
            }
        }
    }

    /// Extract flat clusters from the ordering using the threshold.
    fn extract_clusters(
        &self,
        ordering: &[OpticsPoint],
        num_nodes: usize,
    ) -> (Vec<Vec<usize>>, Vec<usize>) {
        let threshold_dist = 1.0 - self.extract_threshold;
        let mut labels = vec![-1i32; num_nodes];
        let mut cluster_id = 0i32;

        for point in ordering {
            let reach = point.reachability.unwrap_or(f64::INFINITY);
            if reach > threshold_dist {
                // Start of a new cluster or noise
                if !point.core_distance.is_some_and(|cd| cd <= threshold_dist) {
                    // Noise
                    continue;
                }
                // New cluster
                cluster_id += 1;
                labels[point.index] = cluster_id - 1;
            } else {
                // Continue current cluster
                labels[point.index] = (cluster_id - 1).max(0);
            }
        }

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

        (clusters, noise)
    }
}

/// Priority queue element (min-heap via reverse comparison).
#[derive(Debug, Clone)]
struct OrderedPoint {
    index: usize,
    distance: f64,
}

impl PartialEq for OrderedPoint {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for OrderedPoint {}

impl PartialOrd for OrderedPoint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_covers_all_nodes() {
        let sims = vec![(0, 1, 0.9), (1, 2, 0.9), (0, 2, 0.85)];
        let optics = Optics::new(2, 0.8);
        let result = optics.cluster(3, &sims);
        assert_eq!(result.ordering.len(), 3);
    }

    #[test]
    fn finds_cluster() {
        let sims = vec![
            (0, 1, 0.95),
            (1, 2, 0.95),
            (0, 2, 0.9),
            (3, 4, 0.95),
            (4, 5, 0.95),
            (3, 5, 0.9),
        ];
        let optics = Optics::new(2, 0.8);
        let result = optics.cluster(6, &sims);
        assert!(!result.clusters.is_empty());
    }

    #[test]
    fn empty() {
        let optics = Optics::new(2, 0.8);
        let result = optics.cluster(0, &[]);
        assert!(result.clusters.is_empty());
    }

    #[test]
    fn all_disconnected() {
        let optics = Optics::new(2, 0.8);
        let result = optics.cluster(5, &[]);
        assert!(result.clusters.is_empty());
        assert_eq!(result.noise.len(), 5);
    }
}
