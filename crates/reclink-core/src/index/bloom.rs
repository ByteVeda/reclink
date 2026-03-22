//! Bloom filter for fast probabilistic set membership testing.
//!
//! Uses the Kirsch-Mitzenmacker double hashing technique to derive k hash
//! functions from two base hashes. No false negatives; configurable false
//! positive rate.

use ahash::AHasher;
use std::hash::{Hash, Hasher};

/// A space-efficient probabilistic set for fast membership testing.
///
/// Supports `insert` and `contains` with no false negatives.
/// The false positive rate is configurable at construction time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
    count: usize,
}

impl BloomFilter {
    /// Creates a new Bloom filter sized for the expected number of items
    /// and desired false positive rate.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is 0 or `fp_rate` is not in (0, 1).
    #[must_use]
    pub fn with_capacity(expected_items: usize, fp_rate: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(fp_rate > 0.0 && fp_rate < 1.0, "fp_rate must be in (0, 1)");

        let n = expected_items as f64;
        let ln2 = std::f64::consts::LN_2;
        // m = -n * ln(p) / ln(2)^2
        let num_bits = (-(n * fp_rate.ln()) / (ln2 * ln2)).ceil() as usize;
        let num_bits = num_bits.max(64); // at least one u64
                                         // k = (m/n) * ln(2)
        let num_hashes = ((num_bits as f64 / n) * ln2).ceil() as usize;
        let num_hashes = num_hashes.max(1);

        let num_words = num_bits.div_ceil(64);
        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            count: 0,
        }
    }

    /// Inserts an item into the filter.
    pub fn insert(&mut self, item: &str) {
        let (h1, h2) = self.hash_pair(item);
        for i in 0..self.num_hashes {
            let idx = self.bit_index(h1, h2, i);
            self.bits[idx / 64] |= 1u64 << (idx % 64);
        }
        self.count += 1;
    }

    /// Tests whether an item may be in the set.
    ///
    /// Returns `false` if the item is definitely not in the set.
    /// May return `true` for items not in the set (false positive).
    #[must_use]
    pub fn contains(&self, item: &str) -> bool {
        let (h1, h2) = self.hash_pair(item);
        for i in 0..self.num_hashes {
            let idx = self.bit_index(h1, h2, i);
            if self.bits[idx / 64] & (1u64 << (idx % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Returns the number of items inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if no items have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the approximate memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.bits.len() * 8 + std::mem::size_of::<Self>()
    }

    /// Returns the theoretical false positive rate given current count.
    #[must_use]
    pub fn estimated_fp_rate(&self) -> f64 {
        let k = self.num_hashes as f64;
        let m = self.num_bits as f64;
        let n = self.count as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    fn hash_pair(&self, item: &str) -> (u64, u64) {
        let mut h1 = AHasher::default();
        item.hash(&mut h1);
        let hash1 = h1.finish();

        let mut h2 = AHasher::default();
        hash1.hash(&mut h2);
        let hash2 = h2.finish();

        (hash1, hash2)
    }

    fn bit_index(&self, h1: u64, h2: u64, i: usize) -> usize {
        (h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.num_bits as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_contains() {
        let mut bf = BloomFilter::with_capacity(100, 0.01);
        bf.insert("hello");
        bf.insert("world");
        assert!(bf.contains("hello"));
        assert!(bf.contains("world"));
        assert_eq!(bf.len(), 2);
    }

    #[test]
    fn no_false_negatives() {
        let mut bf = BloomFilter::with_capacity(1000, 0.01);
        let items: Vec<String> = (0..500).map(|i| format!("item_{i}")).collect();
        for item in &items {
            bf.insert(item);
        }
        for item in &items {
            assert!(bf.contains(item), "False negative for {item}");
        }
    }

    #[test]
    fn false_positive_rate_bounded() {
        let mut bf = BloomFilter::with_capacity(1000, 0.05);
        for i in 0..1000 {
            bf.insert(&format!("inserted_{i}"));
        }
        let mut fps = 0;
        let trials = 10000;
        for i in 0..trials {
            if bf.contains(&format!("not_inserted_{i}")) {
                fps += 1;
            }
        }
        let rate = fps as f64 / trials as f64;
        assert!(rate < 0.10, "FP rate {rate} too high");
    }

    #[test]
    fn empty_filter() {
        let bf = BloomFilter::with_capacity(100, 0.01);
        assert!(!bf.contains("anything"));
        assert!(bf.is_empty());
    }

    #[test]
    fn memory_usage_reasonable() {
        let bf = BloomFilter::with_capacity(10000, 0.01);
        assert!(bf.memory_usage() < 200_000); // ~12KB expected for 10K items at 1%
    }
}
