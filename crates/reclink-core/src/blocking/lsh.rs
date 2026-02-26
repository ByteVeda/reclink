//! Locality-Sensitive Hashing (LSH) blocking using MinHash.

use ahash::{AHashMap, AHashSet};

use crate::blocking::BlockingStrategy;
use crate::record::{CandidatePair, RecordBatch};

/// LSH blocking uses MinHash signatures and banding to find approximately
/// similar records efficiently.
#[derive(Debug, Clone)]
pub struct LshBlocking {
    /// The field to compute MinHash on.
    pub field: String,
    /// Number of hash functions (signature length).
    pub num_hashes: usize,
    /// Number of bands for banding technique.
    pub num_bands: usize,
    /// Size of character n-grams for shingling.
    pub shingle_size: usize,
}

impl LshBlocking {
    /// Creates a new LSH blocker.
    #[must_use]
    pub fn new(field: impl Into<String>, num_hashes: usize, num_bands: usize) -> Self {
        Self {
            field: field.into(),
            num_hashes,
            num_bands,
            shingle_size: 3,
        }
    }
}

/// Computes shingles (character n-grams) of a string.
fn shingle(s: &str, k: usize) -> AHashSet<String> {
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
fn minhash_signature(shingles: &AHashSet<String>, num_hashes: usize) -> Vec<u64> {
    let mut signature = vec![u64::MAX; num_hashes];

    for shingle in shingles {
        for (i, sig) in signature.iter_mut().enumerate() {
            // Use different hash seeds for each function
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
fn band_keys(signature: &[u64], num_bands: usize) -> Vec<Vec<u64>> {
    let rows_per_band = signature.len() / num_bands;
    if rows_per_band == 0 {
        return vec![signature.to_vec()];
    }
    signature
        .chunks(rows_per_band)
        .map(|chunk| chunk.to_vec())
        .collect()
}

impl BlockingStrategy for LshBlocking {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let signatures: Vec<Option<Vec<u64>>> = records
            .records
            .iter()
            .map(|r| {
                r.get_text(&self.field).map(|v| {
                    let shingles = shingle(&v.to_lowercase(), self.shingle_size);
                    minhash_signature(&shingles, self.num_hashes)
                })
            })
            .collect();

        let mut band_buckets: AHashMap<(usize, Vec<u64>), Vec<usize>> = AHashMap::new();
        for (i, sig) in signatures.iter().enumerate() {
            if let Some(sig) = sig {
                for (band_idx, key) in band_keys(sig, self.num_bands).into_iter().enumerate() {
                    band_buckets.entry((band_idx, key)).or_default().push(i);
                }
            }
        }

        let mut seen = AHashSet::new();
        let mut pairs = Vec::new();
        for indices in band_buckets.values() {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let key = (indices[i].min(indices[j]), indices[i].max(indices[j]));
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
        let left_sigs: Vec<Option<Vec<u64>>> = left
            .records
            .iter()
            .map(|r| {
                r.get_text(&self.field).map(|v| {
                    let shingles = shingle(&v.to_lowercase(), self.shingle_size);
                    minhash_signature(&shingles, self.num_hashes)
                })
            })
            .collect();

        let right_sigs: Vec<Option<Vec<u64>>> = right
            .records
            .iter()
            .map(|r| {
                r.get_text(&self.field).map(|v| {
                    let shingles = shingle(&v.to_lowercase(), self.shingle_size);
                    minhash_signature(&shingles, self.num_hashes)
                })
            })
            .collect();

        // Index right dataset bands
        let mut right_bands: AHashMap<(usize, Vec<u64>), Vec<usize>> = AHashMap::new();
        for (i, sig) in right_sigs.iter().enumerate() {
            if let Some(sig) = sig {
                for (band_idx, key) in band_keys(sig, self.num_bands).into_iter().enumerate() {
                    right_bands.entry((band_idx, key)).or_default().push(i);
                }
            }
        }

        let mut seen = AHashSet::new();
        let mut pairs = Vec::new();
        for (i, sig) in left_sigs.iter().enumerate() {
            if let Some(sig) = sig {
                for (band_idx, key) in band_keys(sig, self.num_bands).into_iter().enumerate() {
                    if let Some(matches) = right_bands.get(&(band_idx, key)) {
                        for &j in matches {
                            if seen.insert((i, j)) {
                                pairs.push(CandidatePair { left: i, right: j });
                            }
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
    fn lsh_dedup_similar() {
        let batch = RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("Jonathan Smith".into())),
                Record::new("2").with_field("name", FieldValue::Text("Jonathon Smith".into())),
                Record::new("3").with_field("name", FieldValue::Text("Xyz Abc".into())),
            ],
        );
        let blocker = LshBlocking::new("name", 100, 20);
        let pairs = blocker.block_dedup(&batch);
        // Jonathan and Jonathon should be similar enough to share a band
        assert!(pairs
            .iter()
            .any(|p| { (p.left == 0 && p.right == 1) || (p.left == 1 && p.right == 0) }));
    }
}
