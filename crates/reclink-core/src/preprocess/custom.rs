//! Custom user-defined preprocessor via a global registry.

use std::sync::{Arc, LazyLock, RwLock};

use ahash::AHashMap;

use crate::error::{ReclinkError, Result};

/// A custom preprocessing function: `(input) -> output`.
pub type CustomPreprocessFn = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Global registry for custom user-defined preprocessors.
static CUSTOM_PREPROCESSORS: LazyLock<RwLock<AHashMap<String, CustomPreprocessFn>>> =
    LazyLock::new(|| RwLock::new(AHashMap::new()));

/// Register a custom preprocessor function under the given name.
///
/// # Errors
///
/// Returns an error if the name is empty.
pub fn register_custom_preprocessor(name: &str, func: CustomPreprocessFn) -> Result<()> {
    if name.is_empty() {
        return Err(ReclinkError::InvalidConfig(
            "preprocessor name cannot be empty".into(),
        ));
    }
    let mut registry = CUSTOM_PREPROCESSORS.write().unwrap();
    registry.insert(name.to_string(), func);
    Ok(())
}

/// Unregister a custom preprocessor. Returns `true` if it existed.
pub fn unregister_custom_preprocessor(name: &str) -> bool {
    let mut registry = CUSTOM_PREPROCESSORS.write().unwrap();
    registry.remove(name).is_some()
}

/// List all registered custom preprocessor names.
#[must_use]
pub fn list_custom_preprocessors() -> Vec<String> {
    let registry = CUSTOM_PREPROCESSORS.read().unwrap();
    registry.keys().cloned().collect()
}

/// Apply a custom preprocessor by name to the given input.
///
/// # Errors
///
/// Returns an error if no preprocessor is registered under the given name.
pub fn apply_custom_preprocessor(name: &str, input: &str) -> Result<String> {
    let registry = CUSTOM_PREPROCESSORS.read().unwrap();
    if let Some(func) = registry.get(name) {
        Ok(func(input))
    } else {
        Err(ReclinkError::InvalidConfig(format!(
            "unknown custom preprocessor: `{name}`"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_use_custom_preprocessor() {
        let name = "test_prep_reg";
        let func: CustomPreprocessFn = Arc::new(|s| s.to_uppercase());

        register_custom_preprocessor(name, func).unwrap();

        let result = apply_custom_preprocessor(name, "hello").unwrap();
        assert_eq!(result, "HELLO");

        assert!(list_custom_preprocessors().contains(&name.to_string()));
        assert!(unregister_custom_preprocessor(name));
        assert!(!unregister_custom_preprocessor(name));
        assert!(apply_custom_preprocessor(name, "test").is_err());
    }
}
