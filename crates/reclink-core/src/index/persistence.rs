//! Serialization and deserialization of index structures.
//!
//! Uses bincode for efficient binary encoding. All index types
//! (`BkTree`, `VpTree`, `NgramIndex`) support save/load.

use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::{ReclinkError, Result};

/// Serialize an index to bytes using bincode.
///
/// # Errors
///
/// Returns an error if serialization fails.
pub fn serialize_to_bytes<T: Serialize>(index: &T) -> Result<Vec<u8>> {
    bincode::serialize(index)
        .map_err(|e| ReclinkError::InvalidConfig(format!("serialization failed: {e}")))
}

/// Deserialize an index from bytes using bincode.
///
/// # Errors
///
/// Returns an error if deserialization fails.
pub fn deserialize_from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    bincode::deserialize(bytes)
        .map_err(|e| ReclinkError::InvalidConfig(format!("deserialization failed: {e}")))
}

/// Save an index to a file.
///
/// # Errors
///
/// Returns an error if serialization or file writing fails.
pub fn save_to_file<T: Serialize>(index: &T, path: &Path) -> Result<()> {
    let bytes = serialize_to_bytes(index)?;
    std::fs::write(path, bytes)
        .map_err(|e| ReclinkError::InvalidConfig(format!("failed to write file: {e}")))
}

/// Load an index from a file.
///
/// # Errors
///
/// Returns an error if file reading or deserialization fails.
pub fn load_from_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path)
        .map_err(|e| ReclinkError::InvalidConfig(format!("failed to read file: {e}")))?;
    deserialize_from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{BkTree, NgramIndex, VpTree};
    use crate::metrics::{JaroWinkler, Levenshtein, Metric};

    #[test]
    fn bk_tree_round_trip() {
        let tree = BkTree::build(
            &["hello", "hallo", "world"],
            Metric::Levenshtein(Levenshtein),
        )
        .unwrap();
        let bytes = serialize_to_bytes(&tree).unwrap();
        let restored: BkTree = deserialize_from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 3);
        let results = restored.find_within("hello", 1);
        assert!(results.iter().any(|r| r.value == "hello"));
        assert!(results.iter().any(|r| r.value == "hallo"));
    }

    #[test]
    fn vp_tree_round_trip() {
        let tree = VpTree::build(
            &["hello", "hallo", "world"],
            Metric::JaroWinkler(JaroWinkler::default()),
        );
        let bytes = serialize_to_bytes(&tree).unwrap();
        let restored: VpTree = deserialize_from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 3);
        let results = restored.find_within("hello", 0.2);
        assert!(results.iter().any(|r| r.value == "hello"));
    }

    #[test]
    fn ngram_index_round_trip() {
        let index = NgramIndex::build(&["hello", "help", "world"], 2);
        let bytes = serialize_to_bytes(&index).unwrap();
        let restored: NgramIndex = deserialize_from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 3);
        let results = restored.search("hello", 2);
        assert!(results.iter().any(|r| r.value == "hello"));
    }

    #[test]
    fn file_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_reclink_persistence.bin");

        let tree = BkTree::build(&["hello", "world"], Metric::Levenshtein(Levenshtein)).unwrap();
        save_to_file(&tree, &path).unwrap();
        let restored: BkTree = load_from_file(&path).unwrap();
        assert_eq!(restored.len(), 2);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
