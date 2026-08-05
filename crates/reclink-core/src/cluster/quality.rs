//! Cluster quality metrics.
//!
//! Measures clustering quality using pairwise similarity data.

use ahash::AHashMap;

/// Computes the silhouette score for a clustering.
///
/// For each point, measures how well it fits its own cluster compared to
/// the nearest other cluster. Returns a score in \[-1, 1\] where 1 means
/// well-clustered and -1 means misclassified.
///
/// Missing pairwise similarities default to distance 1.0.
#[must_use]
pub fn silhouette_score(
    num_nodes: usize,
    similarities: &[(usize, usize, f64)],
    labels: &[i32],
) -> f64 {
    if num_nodes < 2 || labels.is_empty() {
        return 0.0;
    }

    // Build distance lookup
    let mut dist_map: AHashMap<(usize, usize), f64> = AHashMap::new();
    for &(a, b, sim) in similarities {
        let d = 1.0 - sim;
        let key = (a.min(b), a.max(b));
        dist_map.insert(key, d);
    }

    // Group nodes by cluster
    let mut clusters: AHashMap<i32, Vec<usize>> = AHashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            clusters.entry(label).or_default().push(i);
        }
    }

    if clusters.len() < 2 {
        return 0.0; // need at least 2 clusters
    }

    let mut total = 0.0;
    let mut count = 0;

    for (i, &label_i) in labels.iter().enumerate() {
        if label_i < 0 {
            continue; // skip noise
        }

        let own_cluster = &clusters[&label_i];
        if own_cluster.len() <= 1 {
            continue; // singleton cluster
        }

        // a(i) = avg distance to own cluster
        let a_i: f64 = own_cluster
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| {
                let key = (i.min(j), i.max(j));
                *dist_map.get(&key).unwrap_or(&1.0)
            })
            .sum::<f64>()
            / (own_cluster.len() - 1) as f64;

        // b(i) = min avg distance to any other cluster
        let b_i = clusters
            .iter()
            .filter(|(&label, _)| label != label_i)
            .map(|(_, members)| {
                if members.is_empty() {
                    return f64::INFINITY;
                }
                members
                    .iter()
                    .map(|&j| {
                        let key = (i.min(j), i.max(j));
                        *dist_map.get(&key).unwrap_or(&1.0)
                    })
                    .sum::<f64>()
                    / members.len() as f64
            })
            .fold(f64::INFINITY, f64::min);

        let max_ab = a_i.max(b_i);
        if max_ab > 0.0 {
            total += (b_i - a_i) / max_ab;
        }
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Computes the Davies-Bouldin index for a clustering.
///
/// Lower values indicate better clustering. Uses average within-cluster
/// pairwise distance as scatter and average inter-cluster distance as
/// separation.
#[must_use]
pub fn davies_bouldin_index(
    num_nodes: usize,
    similarities: &[(usize, usize, f64)],
    labels: &[i32],
) -> f64 {
    if num_nodes < 2 || labels.is_empty() {
        return 0.0;
    }

    let mut dist_map: AHashMap<(usize, usize), f64> = AHashMap::new();
    for &(a, b, sim) in similarities {
        let key = (a.min(b), a.max(b));
        dist_map.insert(key, 1.0 - sim);
    }

    let mut clusters: AHashMap<i32, Vec<usize>> = AHashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            clusters.entry(label).or_default().push(i);
        }
    }

    let cluster_ids: Vec<i32> = clusters.keys().copied().collect();
    let k = cluster_ids.len();
    if k < 2 {
        return 0.0;
    }

    // Compute scatter (avg within-cluster distance) for each cluster
    let scatter: AHashMap<i32, f64> = cluster_ids
        .iter()
        .map(|&id| {
            let members = &clusters[&id];
            if members.len() <= 1 {
                return (id, 0.0);
            }
            let mut total = 0.0;
            let mut pairs = 0;
            for (ai, &a) in members.iter().enumerate() {
                for &b in &members[ai + 1..] {
                    let key = (a.min(b), a.max(b));
                    total += dist_map.get(&key).unwrap_or(&1.0);
                    pairs += 1;
                }
            }
            (id, if pairs > 0 { total / pairs as f64 } else { 0.0 })
        })
        .collect();

    // Compute inter-cluster distances
    let mut db_sum = 0.0;
    for (ci_idx, &ci) in cluster_ids.iter().enumerate() {
        let mut max_ratio = 0.0f64;
        for (cj_idx, &cj) in cluster_ids.iter().enumerate() {
            if ci_idx == cj_idx {
                continue;
            }
            let members_i = &clusters[&ci];
            let members_j = &clusters[&cj];

            // Inter-cluster distance: avg distance between members
            let mut total_dist = 0.0;
            let mut count = 0;
            for &a in members_i {
                for &b in members_j {
                    let key = (a.min(b), a.max(b));
                    total_dist += dist_map.get(&key).unwrap_or(&1.0);
                    count += 1;
                }
            }
            let inter_dist = if count > 0 {
                total_dist / count as f64
            } else {
                1.0
            };

            if inter_dist > 0.0 {
                let ratio = (scatter[&ci] + scatter[&cj]) / inter_dist;
                max_ratio = max_ratio.max(ratio);
            }
        }
        db_sum += max_ratio;
    }

    db_sum / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silhouette_perfect_clustering() {
        // Two well-separated clusters
        let sims = vec![
            (0, 1, 0.95),
            (2, 3, 0.95),
            (0, 2, 0.1),
            (0, 3, 0.1),
            (1, 2, 0.1),
            (1, 3, 0.1),
        ];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(4, &sims, &labels);
        assert!(score > 0.5, "Expected good silhouette, got {score}");
    }

    #[test]
    fn silhouette_single_cluster() {
        let labels = vec![0, 0, 0];
        let score = silhouette_score(3, &[], &labels);
        assert_eq!(score, 0.0); // need at least 2 clusters
    }

    #[test]
    fn davies_bouldin_well_separated() {
        let sims = vec![
            (0, 1, 0.95),
            (2, 3, 0.95),
            (0, 2, 0.1),
            (0, 3, 0.1),
            (1, 2, 0.1),
            (1, 3, 0.1),
        ];
        let labels = vec![0, 0, 1, 1];
        let db = davies_bouldin_index(4, &sims, &labels);
        assert!(db < 1.0, "Expected low DB index, got {db}");
    }

    #[test]
    fn empty() {
        assert_eq!(silhouette_score(0, &[], &[]), 0.0);
        assert_eq!(davies_bouldin_index(0, &[], &[]), 0.0);
    }
}
