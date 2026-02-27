//! MinHash/LSH index for approximate nearest-neighbor search.
//!
//! Provides a standalone index that uses MinHash signatures and banding
//! for efficient approximate similarity search over large string collections.

use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Serialize};

/// Computes character n-gram shingles of a string.
pub fn shingle(s: &str, k: usize) -> AHashSet<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < k {
        let mut set = AHashSet::new();
        set.insert(s.to_string());
        return set;
    }
    chars
        .windows(k)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

/// Computes a MinHash signature for a set of shingles.
pub fn minhash_signature(shingles: &AHashSet<String>, num_hashes: usize) -> Vec<u64> {
    let mut signature = vec![u64::MAX; num_hashes];
    for shingle in shingles {
        for (i, sig) in signature.iter_mut().enumerate() {
            let hash = ahash::RandomState::with_seeds(
                i as u64,
                i as u64 * 31,
                i as u64 * 37,
                i as u64 * 41,
            )
            .hash_one(shingle);
            *sig = (*sig).min(hash);
        }
    }
    signature
}

/// Extracts band keys from a MinHash signature.
pub fn band_keys(signature: &[u64], num_bands: usize) -> Vec<Vec<u64>> {
    let rows_per_band = signature.len() / num_bands;
    if rows_per_band == 0 {
        return vec![signature.to_vec()];
    }
    signature
        .chunks(rows_per_band)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Estimates Jaccard similarity from two MinHash signatures.
#[must_use]
pub fn estimate_similarity(sig_a: &[u64], sig_b: &[u64]) -> f64 {
    if sig_a.len() != sig_b.len() || sig_a.is_empty() {
        return 0.0;
    }
    let matches = sig_a
        .iter()
        .zip(sig_b.iter())
        .filter(|(a, b)| a == b)
        .count();
    matches as f64 / sig_a.len() as f64
}

/// A standalone MinHash/LSH index for approximate string similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinHashIndex {
    /// Original strings.
    strings: Vec<String>,
    /// Precomputed MinHash signatures.
    signatures: Vec<Vec<u64>>,
    /// Band buckets: (band_index, band_key) → list of string indices.
    #[serde(skip)]
    band_buckets: AHashMap<(usize, Vec<u64>), Vec<usize>>,
    /// Number of hash functions (signature length).
    num_hashes: usize,
    /// Number of bands for the banding technique.
    num_bands: usize,
    /// Shingle size for character n-grams.
    shingle_size: usize,
    /// Deleted indices.
    deleted: AHashSet<usize>,
}

impl MinHashIndex {
    /// Build an index from a collection of strings.
    #[must_use]
    pub fn build(strings: &[&str], num_hashes: usize, num_bands: usize) -> Self {
        Self::build_with_shingle_size(strings, num_hashes, num_bands, 3)
    }

    /// Build an index with a custom shingle size.
    #[must_use]
    pub fn build_with_shingle_size(
        strings: &[&str],
        num_hashes: usize,
        num_bands: usize,
        shingle_size: usize,
    ) -> Self {
        let owned: Vec<String> = strings.iter().map(|s| s.to_string()).collect();
        let signatures: Vec<Vec<u64>> = owned
            .iter()
            .map(|s| {
                let shingles = shingle(&s.to_lowercase(), shingle_size);
                minhash_signature(&shingles, num_hashes)
            })
            .collect();

        let mut band_buckets: AHashMap<(usize, Vec<u64>), Vec<usize>> = AHashMap::new();
        for (i, sig) in signatures.iter().enumerate() {
            for (band_idx, key) in band_keys(sig, num_bands).into_iter().enumerate() {
                band_buckets.entry((band_idx, key)).or_default().push(i);
            }
        }

        Self {
            strings: owned,
            signatures,
            band_buckets,
            num_hashes,
            num_bands,
            shingle_size,
            deleted: AHashSet::new(),
        }
    }

    /// Query for similar strings. Returns indices and estimated similarities
    /// for all candidates above the threshold.
    #[must_use]
    pub fn query(&self, s: &str, threshold: f64) -> Vec<(usize, String, f64)> {
        let shingles = shingle(&s.to_lowercase(), self.shingle_size);
        let query_sig = minhash_signature(&shingles, self.num_hashes);

        let mut candidate_set = AHashSet::new();
        for (band_idx, key) in band_keys(&query_sig, self.num_bands)
            .into_iter()
            .enumerate()
        {
            if let Some(indices) = self.band_buckets.get(&(band_idx, key)) {
                for &idx in indices {
                    if !self.deleted.contains(&idx) {
                        candidate_set.insert(idx);
                    }
                }
            }
        }

        let mut results: Vec<(usize, String, f64)> = candidate_set
            .into_iter()
            .filter_map(|idx| {
                let sim = estimate_similarity(&query_sig, &self.signatures[idx]);
                if sim >= threshold {
                    Some((idx, self.strings[idx].clone(), sim))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Insert a new string into the index. Returns the assigned index.
    pub fn insert(&mut self, s: &str) -> usize {
        let idx = self.strings.len();
        self.strings.push(s.to_string());

        let shingles = shingle(&s.to_lowercase(), self.shingle_size);
        let sig = minhash_signature(&shingles, self.num_hashes);

        for (band_idx, key) in band_keys(&sig, self.num_bands).into_iter().enumerate() {
            self.band_buckets
                .entry((band_idx, key))
                .or_default()
                .push(idx);
        }

        self.signatures.push(sig);
        idx
    }

    /// Soft-delete a string by index.
    pub fn remove(&mut self, idx: usize) -> bool {
        if idx < self.strings.len() && !self.deleted.contains(&idx) {
            self.deleted.insert(idx);
            true
        } else {
            false
        }
    }

    /// Number of active (non-deleted) strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strings.len() - self.deleted.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rebuild the band buckets (e.g. after deserialization).
    pub fn rebuild_buckets(&mut self) {
        self.band_buckets.clear();
        for (i, sig) in self.signatures.iter().enumerate() {
            if !self.deleted.contains(&i) {
                for (band_idx, key) in band_keys(sig, self.num_bands).into_iter().enumerate() {
                    self.band_buckets
                        .entry((band_idx, key))
                        .or_default()
                        .push(i);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_query() {
        let strings = vec!["Jonathan Smith", "Jonathon Smith", "Xyz Abc"];
        let index = MinHashIndex::build(&strings, 100, 20);
        assert_eq!(index.len(), 3);

        let results = index.query("Jonathan Smith", 0.3);
        assert!(!results.is_empty());
        // The exact match should be found
        assert!(results.iter().any(|(_, s, _)| s == "Jonathan Smith"));
    }

    #[test]
    fn similar_strings_found() {
        let strings = vec!["Jonathan Smith", "Jonathon Smith", "Xyz Abc"];
        let index = MinHashIndex::build(&strings, 100, 20);

        let results = index.query("Jonathan Smith", 0.3);
        // Both similar strings should be candidates
        let found_names: Vec<&str> = results.iter().map(|(_, s, _)| s.as_str()).collect();
        assert!(found_names.contains(&"Jonathan Smith"));
        assert!(found_names.contains(&"Jonathon Smith"));
    }

    #[test]
    fn insert_and_query() {
        let mut index = MinHashIndex::build(&["hello world"], 50, 10);
        assert_eq!(index.len(), 1);

        let idx = index.insert("hello worl");
        assert_eq!(idx, 1);
        assert_eq!(index.len(), 2);

        let results = index.query("hello world", 0.3);
        assert!(results.len() >= 1);
    }

    #[test]
    fn remove_and_query() {
        let strings = vec!["abc def", "abc deg", "xyz"];
        let mut index = MinHashIndex::build(&strings, 50, 10);
        assert_eq!(index.len(), 3);

        assert!(index.remove(0));
        assert_eq!(index.len(), 2);

        let results = index.query("abc def", 0.3);
        assert!(results.iter().all(|(idx, _, _)| *idx != 0));
    }

    #[test]
    fn estimate_similarity_identical() {
        let sig = vec![1, 2, 3, 4, 5];
        assert!((estimate_similarity(&sig, &sig) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_similarity_different() {
        let sig_a = vec![1, 2, 3, 4, 5];
        let sig_b = vec![6, 7, 8, 9, 10];
        assert_eq!(estimate_similarity(&sig_a, &sig_b), 0.0);
    }

    #[test]
    fn empty_index() {
        let index = MinHashIndex::build(&[], 50, 10);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let results = index.query("anything", 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn shingle_basic() {
        let shingles = shingle("hello", 3);
        assert!(shingles.contains("hel"));
        assert!(shingles.contains("ell"));
        assert!(shingles.contains("llo"));
    }
}
