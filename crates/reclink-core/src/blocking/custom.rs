//! Custom user-defined blocking strategy via a global registry.

use std::sync::{Arc, LazyLock, RwLock};

use ahash::AHashMap;

use crate::blocking::BlockingStrategy;
use crate::error::{ReclinkError, Result};
use crate::record::{CandidatePair, RecordBatch};

/// A custom dedup blocking function: `(records) -> candidate pairs`.
pub type CustomBlockerDedupFn = Arc<dyn Fn(&RecordBatch) -> Vec<CandidatePair> + Send + Sync>;

/// A custom link blocking function: `(left, right) -> candidate pairs`.
pub type CustomBlockerLinkFn =
    Arc<dyn Fn(&RecordBatch, &RecordBatch) -> Vec<CandidatePair> + Send + Sync>;

/// A registered custom blocker holding both dedup and link functions.
struct RegisteredBlocker {
    dedup_fn: CustomBlockerDedupFn,
    link_fn: CustomBlockerLinkFn,
}

/// Global registry for custom user-defined blockers.
static CUSTOM_BLOCKERS: LazyLock<RwLock<AHashMap<String, RegisteredBlocker>>> =
    LazyLock::new(|| RwLock::new(AHashMap::new()));

/// Register a custom blocker under the given name.
///
/// # Errors
///
/// Returns an error if the name is empty.
pub fn register_custom_blocker(
    name: &str,
    dedup_fn: CustomBlockerDedupFn,
    link_fn: CustomBlockerLinkFn,
) -> Result<()> {
    if name.is_empty() {
        return Err(ReclinkError::InvalidConfig(
            "blocker name cannot be empty".into(),
        ));
    }
    let mut registry = CUSTOM_BLOCKERS.write().unwrap();
    registry.insert(name.to_string(), RegisteredBlocker { dedup_fn, link_fn });
    Ok(())
}

/// Unregister a custom blocker. Returns `true` if it existed.
pub fn unregister_custom_blocker(name: &str) -> bool {
    let mut registry = CUSTOM_BLOCKERS.write().unwrap();
    registry.remove(name).is_some()
}

/// List all registered custom blocker names.
#[must_use]
pub fn list_custom_blockers() -> Vec<String> {
    let registry = CUSTOM_BLOCKERS.read().unwrap();
    registry.keys().cloned().collect()
}

/// A custom blocker that delegates to registered functions.
pub struct CustomBlocker {
    name: String,
    dedup_fn: CustomBlockerDedupFn,
    link_fn: CustomBlockerLinkFn,
}

/// Look up and instantiate a custom blocker by name.
///
/// # Errors
///
/// Returns an error if no blocker is registered under the given name.
pub fn custom_blocker_from_name(name: &str) -> Result<CustomBlocker> {
    let registry = CUSTOM_BLOCKERS.read().unwrap();
    if let Some(registered) = registry.get(name) {
        Ok(CustomBlocker {
            name: name.to_string(),
            dedup_fn: Arc::clone(&registered.dedup_fn),
            link_fn: Arc::clone(&registered.link_fn),
        })
    } else {
        Err(ReclinkError::InvalidConfig(format!(
            "unknown custom blocker: `{name}`"
        )))
    }
}

impl BlockingStrategy for CustomBlocker {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        (self.dedup_fn)(records)
    }

    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        (self.link_fn)(left, right)
    }
}

impl std::fmt::Debug for CustomBlocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomBlocker")
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{FieldValue, Record};

    fn make_batch() -> RecordBatch {
        RecordBatch::new(
            vec!["name".to_string()],
            vec![
                Record::new("1").with_field("name", FieldValue::Text("Alice".into())),
                Record::new("2").with_field("name", FieldValue::Text("Bob".into())),
            ],
        )
    }

    #[test]
    fn register_and_use_custom_blocker() {
        let name = "test_blocker_reg";
        let dedup_fn: CustomBlockerDedupFn =
            Arc::new(|_records| vec![CandidatePair { left: 0, right: 1 }]);
        let link_fn: CustomBlockerLinkFn =
            Arc::new(|_left, _right| vec![CandidatePair { left: 0, right: 0 }]);

        register_custom_blocker(name, dedup_fn, link_fn).unwrap();

        let blocker = custom_blocker_from_name(name).unwrap();
        let batch = make_batch();
        let pairs = blocker.block_dedup(&batch);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 1);

        assert!(list_custom_blockers().contains(&name.to_string()));
        assert!(unregister_custom_blocker(name));
        assert!(!unregister_custom_blocker(name));
        assert!(custom_blocker_from_name(name).is_err());
    }
}
