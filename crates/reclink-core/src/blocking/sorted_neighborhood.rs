//! Sorted neighborhood blocking strategy.

use crate::blocking::BlockingStrategy;
use crate::record::{CandidatePair, RecordBatch};

/// Sorts records by a key field and compares each record with its
/// neighbors within a configurable window.
#[derive(Debug, Clone)]
pub struct SortedNeighborhood {
    /// The field name to sort on.
    pub field: String,
    /// Window size (number of neighbors to compare). Default: 3.
    pub window: usize,
}

impl SortedNeighborhood {
    /// Creates a new sorted neighborhood blocker.
    #[must_use]
    pub fn new(field: impl Into<String>, window: usize) -> Self {
        Self {
            field: field.into(),
            window: window.max(2),
        }
    }
}

impl BlockingStrategy for SortedNeighborhood {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let mut indexed: Vec<(usize, String)> = records
            .records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.get_text(&self.field).map(|v| (i, v.to_lowercase())))
            .collect();

        indexed.sort_by(|a, b| a.1.cmp(&b.1));

        let mut pairs = Vec::new();
        for i in 0..indexed.len() {
            for j in (i + 1)..indexed.len().min(i + self.window) {
                pairs.push(CandidatePair {
                    left: indexed[i].0,
                    right: indexed[j].0,
                });
            }
        }
        pairs
    }

    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        // Merge both datasets with a tag, sort, then generate pairs across datasets
        #[derive(Debug)]
        enum Source {
            Left(usize),
            Right(usize),
        }

        let mut all: Vec<(Source, String)> = Vec::new();
        for (i, r) in left.records.iter().enumerate() {
            if let Some(v) = r.get_text(&self.field) {
                all.push((Source::Left(i), v.to_lowercase()));
            }
        }
        for (i, r) in right.records.iter().enumerate() {
            if let Some(v) = r.get_text(&self.field) {
                all.push((Source::Right(i), v.to_lowercase()));
            }
        }

        all.sort_by(|a, b| a.1.cmp(&b.1));

        let mut pairs = Vec::new();
        for i in 0..all.len() {
            for j in (i + 1)..all.len().min(i + self.window) {
                match (&all[i].0, &all[j].0) {
                    (Source::Left(l), Source::Right(r)) => {
                        pairs.push(CandidatePair {
                            left: *l,
                            right: *r,
                        });
                    }
                    (Source::Right(r), Source::Left(l)) => {
                        pairs.push(CandidatePair {
                            left: *l,
                            right: *r,
                        });
                    }
                    _ => {}
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
    fn sorted_neighborhood_dedup() {
        let batch = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("Alice".into())),
                Record::new("2").with_field("name", FieldValue::Text("Bob".into())),
                Record::new("3").with_field("name", FieldValue::Text("Alicia".into())),
            ],
        );
        let blocker = SortedNeighborhood::new("name", 3);
        let pairs = blocker.block_dedup(&batch);
        // After sorting: Alice(0), Alicia(2), Bob(1) - window=3 means all compared
        assert!(!pairs.is_empty());
    }
}
