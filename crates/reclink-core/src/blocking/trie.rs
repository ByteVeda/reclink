//! Trie-based blocking strategy with prefix and frequency filtering.
//!
//! Groups records by shared prefixes using a trie structure. Nodes with
//! frequency exceeding `max_frequency` are pruned to avoid generating
//! excessive candidate pairs from common prefixes (e.g., "Mc", "De").

use ahash::{AHashMap, AHashSet};

use crate::blocking::BlockingStrategy;
use crate::record::{CandidatePair, RecordBatch};

/// A node in the prefix trie.
#[derive(Debug, Clone)]
struct TrieNode {
    /// Children keyed by character.
    children: AHashMap<char, Box<TrieNode>>,
    /// Record indices stored at this node (records whose prefix ends here
    /// or that traverse through this node).
    indices: Vec<usize>,
    /// Total number of records in this subtree.
    frequency: usize,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: AHashMap::new(),
            indices: Vec::new(),
            frequency: 0,
        }
    }

    /// Insert a record index into the trie for the given key.
    fn insert(&mut self, key: &[char], index: usize, depth: usize, min_prefix_len: usize) {
        self.frequency += 1;

        if depth >= min_prefix_len {
            self.indices.push(index);
        }

        if let Some((&first, rest)) = key.split_first() {
            let child = self
                .children
                .entry(first)
                .or_insert_with(|| Box::new(TrieNode::new()));
            child.insert(rest, index, depth + 1, min_prefix_len);
        }
    }

    /// Collect all candidate pairs from this node and its children,
    /// respecting the frequency limit.
    fn collect_pairs(&self, max_frequency: usize, pairs: &mut AHashSet<(usize, usize)>) {
        // Only generate pairs from nodes within the frequency limit
        if self.frequency <= max_frequency && self.indices.len() >= 2 {
            for i in 0..self.indices.len() {
                for j in (i + 1)..self.indices.len() {
                    let (a, b) = if self.indices[i] < self.indices[j] {
                        (self.indices[i], self.indices[j])
                    } else {
                        (self.indices[j], self.indices[i])
                    };
                    pairs.insert((a, b));
                }
            }
        }

        // Recurse into children
        for child in self.children.values() {
            child.collect_pairs(max_frequency, pairs);
        }
    }

    /// Collect candidate pairs for linkage (left vs right records).
    /// Left indices and right indices are stored together but distinguished
    /// by the sets passed in.
    fn collect_link_pairs(
        &self,
        max_frequency: usize,
        left_set: &AHashSet<usize>,
        pairs: &mut AHashSet<(usize, usize)>,
    ) {
        if self.frequency <= max_frequency && self.indices.len() >= 2 {
            let lefts: Vec<usize> = self
                .indices
                .iter()
                .copied()
                .filter(|i| left_set.contains(i))
                .collect();
            let rights: Vec<usize> = self
                .indices
                .iter()
                .copied()
                .filter(|i| !left_set.contains(i))
                .collect();

            for &l in &lefts {
                for &r in &rights {
                    pairs.insert((l, r));
                }
            }
        }

        for child in self.children.values() {
            child.collect_link_pairs(max_frequency, left_set, pairs);
        }
    }
}

/// Trie-based blocking strategy.
///
/// Records are inserted into a character trie by their field value (lowercased).
/// Candidate pairs are generated from records that share a common prefix of at
/// least `min_prefix_len` characters. Trie nodes with more records than
/// `max_frequency` are skipped to prune overly common prefixes.
#[derive(Debug, Clone)]
pub struct TrieBlocking {
    /// The field name to block on.
    pub field: String,
    /// Minimum prefix length before records can be paired.
    pub min_prefix_len: usize,
    /// Maximum number of records in a trie subtree to generate pairs from.
    /// Nodes with more records are pruned.
    pub max_frequency: usize,
}

impl TrieBlocking {
    /// Creates a new trie blocker.
    #[must_use]
    pub fn new(field: impl Into<String>, min_prefix_len: usize, max_frequency: usize) -> Self {
        Self {
            field: field.into(),
            min_prefix_len: min_prefix_len.max(1),
            max_frequency: max_frequency.max(2),
        }
    }
}

impl BlockingStrategy for TrieBlocking {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let mut root = TrieNode::new();

        for (i, record) in records.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                let lower = val.to_lowercase();
                let chars: Vec<char> = lower.chars().collect();
                if chars.len() >= self.min_prefix_len {
                    root.insert(&chars, i, 0, self.min_prefix_len);
                }
            }
        }

        let mut pairs = AHashSet::new();
        root.collect_pairs(self.max_frequency, &mut pairs);

        pairs
            .into_iter()
            .map(|(l, r)| CandidatePair { left: l, right: r })
            .collect()
    }

    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        let mut root = TrieNode::new();
        let mut left_set = AHashSet::new();

        // Insert left records with their original indices
        for (i, record) in left.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                let lower = val.to_lowercase();
                let chars: Vec<char> = lower.chars().collect();
                if chars.len() >= self.min_prefix_len {
                    root.insert(&chars, i, 0, self.min_prefix_len);
                    left_set.insert(i);
                }
            }
        }

        // Insert right records with offset indices
        let offset = left.records.len();
        for (i, record) in right.records.iter().enumerate() {
            if let Some(val) = record.get_text(&self.field) {
                let lower = val.to_lowercase();
                let chars: Vec<char> = lower.chars().collect();
                if chars.len() >= self.min_prefix_len {
                    root.insert(&chars, offset + i, 0, self.min_prefix_len);
                }
            }
        }

        let mut pairs = AHashSet::new();
        root.collect_link_pairs(self.max_frequency, &left_set, &mut pairs);

        pairs
            .into_iter()
            .map(|(l, r)| CandidatePair {
                left: l,
                right: r - offset,
            })
            .collect()
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
                Record::new("2").with_field("name", FieldValue::Text("Smyth".into())),
                Record::new("3").with_field("name", FieldValue::Text("Jones".into())),
                Record::new("4").with_field("name", FieldValue::Text("Johnson".into())),
                Record::new("5").with_field("name", FieldValue::Text("Smithson".into())),
            ],
        )
    }

    #[test]
    fn dedup_pairs_shared_prefix() {
        // min_prefix_len=2 means records sharing 2+ char prefix are paired
        let blocker = TrieBlocking::new("name", 2, 100);
        let batch = make_batch();
        let pairs = blocker.block_dedup(&batch);

        // "smith" and "smyth" and "smithson" share "sm" prefix
        let has_smith_smyth = pairs
            .iter()
            .any(|p| (p.left == 0 && p.right == 1) || (p.left == 1 && p.right == 0));
        assert!(has_smith_smyth, "Smith and Smyth share 'sm' prefix");

        let has_smith_smithson = pairs
            .iter()
            .any(|p| (p.left == 0 && p.right == 4) || (p.left == 4 && p.right == 0));
        assert!(has_smith_smithson, "Smith and Smithson share 'sm' prefix");

        // "jones" and "johnson" share "jo" prefix
        let has_jones_johnson = pairs
            .iter()
            .any(|p| (p.left == 2 && p.right == 3) || (p.left == 3 && p.right == 2));
        assert!(has_jones_johnson, "Jones and Johnson share 'jo' prefix");
    }

    #[test]
    fn frequency_pruning() {
        // Build batch where many records share same prefix
        let records: Vec<Record> = (0..10)
            .map(|i| {
                Record::new(format!("{i}"))
                    .with_field("name", FieldValue::Text(format!("Smith{i}")))
            })
            .collect();
        let batch = RecordBatch::new(vec!["name".to_string()], records);

        // max_frequency=3 should prune the "sm" prefix node (10 records)
        let blocker = TrieBlocking::new("name", 2, 3);
        let pairs = blocker.block_dedup(&batch);
        // All 10 records share "sm" prefix but node has freq 10 > 3, so pruned
        // However, deeper nodes (e.g., "smi", "smit") also have freq 10
        // Only leaf-level unique nodes with freq <= 3 would produce pairs
        assert!(
            pairs.len() < 45,
            "frequency pruning should reduce pairs (got {})",
            pairs.len()
        );
    }

    #[test]
    fn link_pairs() {
        let blocker = TrieBlocking::new("name", 2, 100);
        let left = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("L1").with_field("name", FieldValue::Text("Smith".into())),
                Record::new("L2").with_field("name", FieldValue::Text("Jones".into())),
            ],
        );
        let right = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("R1").with_field("name", FieldValue::Text("Smyth".into())),
                Record::new("R2").with_field("name", FieldValue::Text("Brown".into())),
            ],
        );
        let pairs = blocker.block_link(&left, &right);

        // Smith (left 0) and Smyth (right 0) share "sm" prefix
        let has_match = pairs.iter().any(|p| p.left == 0 && p.right == 0);
        assert!(has_match, "Smith and Smyth should be linked");

        // Jones and Brown should NOT be linked
        let has_no_match = !pairs.iter().any(|p| p.left == 1 && p.right == 1);
        assert!(has_no_match, "Jones and Brown should not be linked");
    }

    #[test]
    fn short_strings_excluded() {
        let blocker = TrieBlocking::new("name", 3, 100);
        let batch = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("AB".into())),
                Record::new("2").with_field("name", FieldValue::Text("AC".into())),
            ],
        );
        let pairs = blocker.block_dedup(&batch);
        assert!(
            pairs.is_empty(),
            "strings shorter than min_prefix_len excluded"
        );
    }

    #[test]
    fn empty_field_skipped() {
        let blocker = TrieBlocking::new("name", 2, 100);
        let batch = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("Smith".into())),
                Record::new("2").with_field("name", FieldValue::Text(String::new())),
            ],
        );
        let pairs = blocker.block_dedup(&batch);
        assert!(pairs.is_empty());
    }
}
