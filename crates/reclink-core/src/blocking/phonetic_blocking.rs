//! Phonetic code blocking strategy.

use ahash::AHashMap;

use crate::blocking::BlockingStrategy;
use crate::phonetic::{PhoneticAlgorithm, PhoneticEncoder, Soundex};
use crate::record::{CandidatePair, RecordBatch};

/// Blocks records by phonetic encoding of a field.
#[derive(Debug, Clone)]
pub struct PhoneticBlocking {
    /// The field name to encode.
    pub field: String,
    /// The phonetic algorithm to use.
    pub algorithm: PhoneticAlgorithm,
}

impl PhoneticBlocking {
    /// Creates a new phonetic blocker with the given algorithm.
    #[must_use]
    pub fn new(field: impl Into<String>, algorithm: PhoneticAlgorithm) -> Self {
        Self {
            field: field.into(),
            algorithm,
        }
    }

    /// Creates a new phonetic blocker using Soundex.
    #[must_use]
    pub fn soundex(field: impl Into<String>) -> Self {
        Self::new(field, PhoneticAlgorithm::Soundex(Soundex))
    }
}

impl BlockingStrategy for PhoneticBlocking {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let mut blocks: AHashMap<String, Vec<usize>> = AHashMap::new();

        for (i, record) in records.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                let code = self.algorithm.encode(val);
                blocks.entry(code).or_default().push(i);
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
                let code = self.algorithm.encode(val);
                right_index.entry(code).or_default().push(i);
            }
        }

        let mut pairs = Vec::new();
        for (i, record) in left.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                let code = self.algorithm.encode(val);
                if let Some(matches) = right_index.get(&code) {
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

    #[test]
    fn phonetic_blocking_dedup() {
        let batch = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("Smith".into())),
                Record::new("2").with_field("name", FieldValue::Text("Smyth".into())),
                Record::new("3").with_field("name", FieldValue::Text("Jones".into())),
            ],
        );
        let blocker = PhoneticBlocking::soundex("name");
        let pairs = blocker.block_dedup(&batch);
        // Smith and Smyth have the same Soundex code (S530)
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 1);
    }
}
