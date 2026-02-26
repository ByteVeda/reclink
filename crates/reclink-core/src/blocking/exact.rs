//! Exact match blocking strategy.

use ahash::AHashMap;

use crate::blocking::BlockingStrategy;
use crate::record::{CandidatePair, RecordBatch};

/// Blocks records by exact equality on a specified field.
#[derive(Debug, Clone)]
pub struct ExactBlocking {
    /// The field name to block on.
    pub field: String,
}

impl ExactBlocking {
    /// Creates a new exact blocker for the given field.
    #[must_use]
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl BlockingStrategy for ExactBlocking {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let mut blocks: AHashMap<String, Vec<usize>> = AHashMap::new();

        for (i, record) in records.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                blocks.entry(val.to_string()).or_default().push(i);
            }
        }

        let mut pairs = Vec::new();
        for indices in blocks.values() {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    pairs.push(CandidatePair {
                        left: indices[i],
                        right: indices[j],
                    });
                }
            }
        }
        pairs
    }

    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        let mut right_index: AHashMap<String, Vec<usize>> = AHashMap::new();
        for (i, record) in right.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                right_index.entry(val.to_string()).or_default().push(i);
            }
        }

        let mut pairs = Vec::new();
        for (i, record) in left.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                if let Some(matches) = right_index.get(val) {
                    for &j in matches {
                        pairs.push(CandidatePair { left: i, right: j });
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

    fn make_batch() -> RecordBatch {
        RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("Smith".into())),
                Record::new("2").with_field("name", FieldValue::Text("Jones".into())),
                Record::new("3").with_field("name", FieldValue::Text("Smith".into())),
            ],
        )
    }

    #[test]
    fn dedup_blocks_matching() {
        let blocker = ExactBlocking::new("name");
        let batch = make_batch();
        let pairs = blocker.block_dedup(&batch);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 2);
    }

    #[test]
    fn link_blocks_matching() {
        let blocker = ExactBlocking::new("name");
        let left = make_batch();
        let right = RecordBatch::new(
            vec!["name".to_string()],
            vec![Record::new("a").with_field("name", FieldValue::Text("Smith".into()))],
        );
        let pairs = blocker.block_link(&left, &right);
        assert_eq!(pairs.len(), 2); // records 0 and 2 match
    }
}
