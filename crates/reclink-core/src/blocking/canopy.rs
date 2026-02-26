//! Canopy clustering blocking strategy.

use crate::blocking::BlockingStrategy;
use crate::metrics::Metric;
use crate::record::{CandidatePair, RecordBatch};

/// Canopy clustering uses a cheap distance metric with two thresholds
/// (tight and loose) to form overlapping clusters.
#[derive(Debug, Clone)]
pub struct CanopyClustering {
    /// The field to compare.
    pub field: String,
    /// Tight threshold: records within this distance are strongly linked.
    pub t_tight: f64,
    /// Loose threshold: records within this distance are candidates.
    pub t_loose: f64,
    /// The metric to use for distance computation.
    pub metric: Metric,
}

impl CanopyClustering {
    /// Creates a new canopy clustering blocker.
    #[must_use]
    pub fn new(field: impl Into<String>, t_tight: f64, t_loose: f64, metric: Metric) -> Self {
        Self {
            field: field.into(),
            t_tight,
            t_loose,
            metric,
        }
    }
}

impl BlockingStrategy for CanopyClustering {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let values: Vec<Option<String>> = records
            .records
            .iter()
            .map(|r| r.get_text(&self.field).map(|s| s.to_lowercase()))
            .collect();

        let n = values.len();
        let mut removed = vec![false; n];
        let mut canopies: Vec<Vec<usize>> = Vec::new();

        for i in 0..n {
            if removed[i] || values[i].is_none() {
                continue;
            }

            let center = values[i].as_ref().unwrap();
            let mut canopy = vec![i];

            for j in (i + 1)..n {
                if removed[j] || values[j].is_none() {
                    continue;
                }

                let other = values[j].as_ref().unwrap();
                let sim = self.metric.similarity(center, other);

                if sim >= self.t_loose {
                    canopy.push(j);
                    if sim >= self.t_tight {
                        removed[j] = true;
                    }
                }
            }

            canopies.push(canopy);
        }

        let mut pairs = Vec::new();
        let mut seen = ahash::AHashSet::new();
        for canopy in &canopies {
            for i in 0..canopy.len() {
                for j in (i + 1)..canopy.len() {
                    let key = (canopy[i].min(canopy[j]), canopy[i].max(canopy[j]));
                    if seen.insert(key) {
                        pairs.push(CandidatePair {
                            left: key.0,
                            right: key.1,
                        });
                    }
                }
            }
        }
        pairs
    }

    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        let left_values: Vec<Option<String>> = left
            .records
            .iter()
            .map(|r| r.get_text(&self.field).map(|s| s.to_lowercase()))
            .collect();

        let right_values: Vec<Option<String>> = right
            .records
            .iter()
            .map(|r| r.get_text(&self.field).map(|s| s.to_lowercase()))
            .collect();

        let mut pairs = Vec::new();
        for (i, left_val) in left_values.iter().enumerate() {
            if let Some(lv) = left_val {
                for (j, right_val) in right_values.iter().enumerate() {
                    if let Some(rv) = right_val {
                        let sim = self.metric.similarity(lv, rv);
                        if sim >= self.t_loose {
                            pairs.push(CandidatePair { left: i, right: j });
                        }
                    }
                }
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{FieldValue, Record, RecordBatch};

    #[test]
    fn canopy_dedup() {
        let batch = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("Smith".into())),
                Record::new("2").with_field("name", FieldValue::Text("Smyth".into())),
                Record::new("3").with_field("name", FieldValue::Text("Jones".into())),
            ],
        );
        let blocker = CanopyClustering::new("name", 0.9, 0.5, Metric::default());
        let pairs = blocker.block_dedup(&batch);
        // Smith and Smyth should be in same canopy
        assert!(pairs
            .iter()
            .any(|p| { (p.left == 0 && p.right == 1) || (p.left == 1 && p.right == 0) }));
    }
}
