//! Q-gram (n-gram) blocking strategy.

use ahash::AHashMap;

use crate::blocking::BlockingStrategy;
use crate::record::{CandidatePair, RecordBatch};

/// Groups records by shared character n-grams, generating candidate pairs
/// for records that share at least one q-gram.
#[derive(Debug, Clone)]
pub struct QgramBlocking {
    /// The field name to extract q-grams from.
    pub field: String,
    /// Size of character n-grams. Default: 3.
    pub q: usize,
    /// Minimum number of shared q-grams to form a pair.
    pub threshold: usize,
}

impl QgramBlocking {
    /// Creates a new q-gram blocker.
    #[must_use]
    pub fn new(field: impl Into<String>, q: usize, threshold: usize) -> Self {
        Self {
            field: field.into(),
            q: q.max(1),
            threshold: threshold.max(1),
        }
    }
}

/// Extracts character q-grams from a string.
fn extract_qgrams(s: &str, q: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < q {
        return vec![s.to_string()];
    }
    chars.windows(q).map(|w| w.iter().collect()).collect()
}

impl BlockingStrategy for QgramBlocking {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let mut index: AHashMap<String, Vec<usize>> = AHashMap::new();

        for (i, record) in records.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                let lower = val.to_lowercase();
                for qg in extract_qgrams(&lower, self.q) {
                    index.entry(qg).or_default().push(i);
                }
            }
        }

        let mut pair_counts: AHashMap<(usize, usize), usize> = AHashMap::new();
        for indices in index.values() {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let key = (indices[i].min(indices[j]), indices[i].max(indices[j]));
                    *pair_counts.entry(key).or_insert(0) += 1;
                }
            }
        }

        pair_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.threshold)
            .map(|((l, r), _)| CandidatePair { left: l, right: r })
            .collect()
    }

    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        let mut right_index: AHashMap<String, Vec<usize>> = AHashMap::new();
        for (i, record) in right.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                let lower = val.to_lowercase();
                for qg in extract_qgrams(&lower, self.q) {
                    right_index.entry(qg).or_default().push(i);
                }
            }
        }

        let mut pair_counts: AHashMap<(usize, usize), usize> = AHashMap::new();
        for (i, record) in left.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                let lower = val.to_lowercase();
                for qg in extract_qgrams(&lower, self.q) {
                    if let Some(matches) = right_index.get(&qg) {
                        for &j in matches {
                            *pair_counts.entry((i, j)).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        pair_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.threshold)
            .map(|((l, r), _)| CandidatePair { left: l, right: r })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{FieldValue, Record, RecordBatch};

    #[test]
    fn qgram_dedup() {
        let batch = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("Smith".into())),
                Record::new("2").with_field("name", FieldValue::Text("Smyth".into())),
                Record::new("3").with_field("name", FieldValue::Text("Jones".into())),
            ],
        );
        let blocker = QgramBlocking::new("name", 2, 1);
        let pairs = blocker.block_dedup(&batch);
        // Smith and Smyth share "th" bigram
        assert!(pairs
            .iter()
            .any(|p| (p.left == 0 && p.right == 1) || (p.left == 1 && p.right == 0)));
    }
}
