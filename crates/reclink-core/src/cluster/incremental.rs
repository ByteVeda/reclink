//! Incremental (streaming) clustering.
//!
//! Assigns records to clusters one at a time using a representative-based
//! approach. Each cluster maintains a representative string, and new records
//! are assigned to the nearest cluster (if within threshold) or create a new
//! cluster.

use crate::metrics::Metric;

/// Assignment result when adding a record to incremental clustering.
#[derive(Debug, Clone)]
pub enum ClusterAssignment {
    /// Record was assigned to an existing cluster.
    Existing {
        /// The cluster ID.
        cluster_id: usize,
        /// Similarity to the cluster representative.
        similarity: f64,
    },
    /// A new cluster was created for this record.
    New {
        /// The new cluster ID.
        cluster_id: usize,
    },
}

/// Incremental clustering that assigns records one at a time.
///
/// Uses a representative-based approach: each cluster has a representative
/// string, and new records are compared against all representatives. The
/// record joins the most similar cluster (if above threshold) or starts a
/// new cluster.
pub struct IncrementalCluster {
    metric: Metric,
    threshold: f64,
    /// Each cluster is a list of record indices.
    clusters: Vec<Vec<usize>>,
    /// Representative string for each cluster (first added record).
    representatives: Vec<String>,
    /// Total records added.
    count: usize,
}

impl IncrementalCluster {
    /// Creates a new incremental clusterer.
    #[must_use]
    pub fn new(metric: Metric, threshold: f64) -> Self {
        Self {
            metric,
            threshold,
            clusters: Vec::new(),
            representatives: Vec::new(),
            count: 0,
        }
    }

    /// Adds a record and assigns it to the best matching cluster or creates
    /// a new one.
    pub fn add_record(&mut self, value: &str) -> ClusterAssignment {
        let record_idx = self.count;
        self.count += 1;

        // Find the most similar representative
        let mut best_sim = 0.0f64;
        let mut best_cluster = None;

        for (i, rep) in self.representatives.iter().enumerate() {
            let sim = self.metric.similarity(value, rep);
            if sim > best_sim {
                best_sim = sim;
                best_cluster = Some(i);
            }
        }

        if best_sim >= self.threshold {
            let cluster_id = best_cluster.unwrap();
            self.clusters[cluster_id].push(record_idx);
            ClusterAssignment::Existing {
                cluster_id,
                similarity: best_sim,
            }
        } else {
            let cluster_id = self.clusters.len();
            self.clusters.push(vec![record_idx]);
            self.representatives.push(value.to_string());
            ClusterAssignment::New { cluster_id }
        }
    }

    /// Returns all clusters as lists of record indices.
    #[must_use]
    pub fn get_clusters(&self) -> &[Vec<usize>] {
        &self.clusters
    }

    /// Returns the number of clusters.
    #[must_use]
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Returns the total number of records added.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.count
    }

    /// Returns the similarity threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::JaroWinkler;

    #[test]
    fn identical_records_same_cluster() {
        let mut ic = IncrementalCluster::new(Metric::JaroWinkler(JaroWinkler::default()), 0.85);

        let a1 = ic.add_record("John Smith");
        assert!(matches!(a1, ClusterAssignment::New { cluster_id: 0 }));

        let a2 = ic.add_record("John Smith");
        assert!(matches!(
            a2,
            ClusterAssignment::Existing { cluster_id: 0, .. }
        ));

        assert_eq!(ic.cluster_count(), 1);
        assert_eq!(ic.get_clusters()[0], vec![0, 1]);
    }

    #[test]
    fn different_records_different_clusters() {
        let mut ic = IncrementalCluster::new(Metric::JaroWinkler(JaroWinkler::default()), 0.85);

        ic.add_record("John Smith");
        ic.add_record("Alice Johnson");

        assert_eq!(ic.cluster_count(), 2);
    }

    #[test]
    fn similar_records_same_cluster() {
        let mut ic = IncrementalCluster::new(Metric::JaroWinkler(JaroWinkler::default()), 0.80);

        ic.add_record("John Smith");
        let assignment = ic.add_record("Jon Smith");

        // "Jon Smith" should be similar enough to "John Smith"
        match assignment {
            ClusterAssignment::Existing {
                cluster_id,
                similarity,
            } => {
                assert_eq!(cluster_id, 0);
                assert!(similarity >= 0.80);
            }
            ClusterAssignment::New { .. } => {
                panic!("Expected existing cluster assignment");
            }
        }
    }

    #[test]
    fn empty_state() {
        let ic = IncrementalCluster::new(Metric::JaroWinkler(JaroWinkler::default()), 0.85);
        assert_eq!(ic.cluster_count(), 0);
        assert_eq!(ic.record_count(), 0);
    }

    #[test]
    fn record_count_tracks() {
        let mut ic = IncrementalCluster::new(Metric::JaroWinkler(JaroWinkler::default()), 0.85);
        ic.add_record("a");
        ic.add_record("b");
        ic.add_record("c");
        assert_eq!(ic.record_count(), 3);
    }
}
