//! Index structures for efficient nearest-neighbor search.

pub mod bk_tree;
pub mod mmap_ngram;
pub mod ngram_index;
pub mod persistence;
pub mod vp_tree;

pub use bk_tree::BkTree;
pub use mmap_ngram::MmapNgramIndex;
pub use ngram_index::NgramIndex;
pub use vp_tree::VpTree;
