//! Blocking strategies for candidate pair generation.
//!
//! Blocking reduces the quadratic comparison space by grouping records
//! that are likely to match into blocks, then only comparing within blocks.

mod canopy;
pub mod custom;
mod date;
mod exact;
mod lsh;
mod numeric;
mod phonetic_blocking;
mod qgram;
mod sorted_neighborhood;
mod trie;

pub use canopy::CanopyClustering;
pub use custom::*;
pub use date::{DateBlocking, DateResolution};
pub use exact::ExactBlocking;
pub use lsh::LshBlocking;
pub use numeric::NumericBlocking;
pub use phonetic_blocking::PhoneticBlocking;
pub use qgram::QgramBlocking;
pub use sorted_neighborhood::SortedNeighborhood;
pub use trie::TrieBlocking;

use crate::record::{CandidatePair, RecordBatch};

/// Trait for blocking strategies that generate candidate pairs.
pub trait BlockingStrategy: Send + Sync {
    /// Generates candidate pairs for deduplication within a single dataset.
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair>;

    /// Generates candidate pairs for linkage between two datasets.
    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair>;
}
