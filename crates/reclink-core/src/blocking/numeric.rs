//! Bucket-based numeric blocking strategy.

use ahash::AHashMap;

use crate::blocking::BlockingStrategy;
use crate::record::{CandidatePair, RecordBatch};

/// Blocks records by bucketing a numeric field into ranges.
///
/// Adjacent buckets also form candidate pairs to handle boundary cases.
#[derive(Debug, Clone)]
pub struct NumericBlocking {
    /// The field name to block on.
    pub field: String,
    /// Width of each bucket (e.g., 5.0 for age ranges 20–24, 25–29).
    pub bucket_size: f64,
}

impl NumericBlocking {
    /// Creates a new numeric blocker.
    #[must_use]
    pub fn new(field: impl Into<String>, bucket_size: f64) -> Self {
        Self {
            field: field.into(),
            bucket_size,
        }
    }

    fn get_bucket(&self, record: &crate::record::Record) -> Option<i64> {
        let val = record.get(&self.field)?.as_f64()?;
        Some((val / self.bucket_size).floor() as i64)
    }
}

impl BlockingStrategy for NumericBlocking {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let mut buckets: AHashMap<i64, Vec<usize>> = AHashMap::new();

        for (i, record) in records.records.iter().enumerate() {
            if let Some(bucket) = self.get_bucket(record) {
                buckets.entry(bucket).or_default().push(i);
            }
        }

        let mut seen = ahash::AHashSet::new();
        let mut pairs = Vec::new();

        // Pairs within the same bucket
        for indices in buckets.values() {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let key = (indices[i], indices[j]);
                    if seen.insert(key) {
                        pairs.push(CandidatePair {
                            left: indices[i],
                            right: indices[j],
                        });
                    }
                }
            }
        }

        // Pairs between adjacent buckets
        let bucket_keys: Vec<i64> = buckets.keys().copied().collect();
        for &bk in &bucket_keys {
            if let Some(adj) = buckets.get(&(bk + 1)) {
                if let Some(cur) = buckets.get(&bk) {
                    for &i in cur {
                        for &j in adj {
                            let key = if i < j { (i, j) } else { (j, i) };
                            if seen.insert(key) {
                                pairs.push(CandidatePair {
                                    left: key.0,
                                    right: key.1,
                                });
                            }
                        }
                    }
                }
            }
        }

        pairs
    }

    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        let mut right_buckets: AHashMap<i64, Vec<usize>> = AHashMap::new();
        for (i, record) in right.records.iter().enumerate() {
            if let Some(bucket) = self.get_bucket(record) {
                right_buckets.entry(bucket).or_default().push(i);
            }
        }

        let mut pairs = Vec::new();
        for (i, record) in left.records.iter().enumerate() {
            if let Some(bucket) = self.get_bucket(record) {
                // Same bucket and adjacent buckets
                for offset in -1..=1 {
                    if let Some(matches) = right_buckets.get(&(bucket + offset)) {
                        for &j in matches {
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

    fn make_batch(values: &[f64]) -> RecordBatch {
        let records: Vec<Record> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                Record::new(format!("{}", i + 1)).with_field("age", FieldValue::Float(v))
            })
            .collect();
        RecordBatch::new(vec!["age".to_string()], records)
    }

    #[test]
    fn numeric_same_bucket() {
        // 22 and 23 are in bucket 4 (22/5=4.4 → 4, 23/5=4.6 → 4)
        let batch = make_batch(&[22.0, 23.0, 50.0]);
        let blocker = NumericBlocking::new("age", 5.0);
        let pairs = blocker.block_dedup(&batch);
        let has_pair = pairs
            .iter()
            .any(|p| (p.left == 0 && p.right == 1) || (p.left == 1 && p.right == 0));
        assert!(has_pair, "22 and 23 should be in the same bucket");
    }

    #[test]
    fn numeric_adjacent_buckets() {
        // 24 is in bucket 4, 25 is in bucket 5 — adjacent
        let batch = make_batch(&[24.0, 25.0]);
        let blocker = NumericBlocking::new("age", 5.0);
        let pairs = blocker.block_dedup(&batch);
        assert!(
            !pairs.is_empty(),
            "24 and 25 should form a pair (adjacent buckets)"
        );
    }

    #[test]
    fn numeric_far_apart() {
        // 10 is in bucket 2, 50 is in bucket 10 — too far
        let batch = make_batch(&[10.0, 50.0]);
        let blocker = NumericBlocking::new("age", 5.0);
        let pairs = blocker.block_dedup(&batch);
        assert!(pairs.is_empty(), "10 and 50 should not form pairs");
    }
}
