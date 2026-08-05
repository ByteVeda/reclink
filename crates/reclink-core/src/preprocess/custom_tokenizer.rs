//! Custom user-defined tokenizer via a global registry.

use std::sync::{Arc, LazyLock, RwLock};

use ahash::AHashMap;

use crate::error::{ReclinkError, Result};

/// A custom tokenizer function: `(input) -> tokens`.
pub type CustomTokenizerFn = Arc<dyn Fn(&str) -> Vec<String> + Send + Sync>;

/// Global registry for custom user-defined tokenizers.
static CUSTOM_TOKENIZERS: LazyLock<RwLock<AHashMap<String, CustomTokenizerFn>>> =
    LazyLock::new(|| RwLock::new(AHashMap::new()));

/// Register a custom tokenizer function under the given name.
///
/// # Errors
///
/// Returns an error if the name is empty.
pub fn register_custom_tokenizer(name: &str, func: CustomTokenizerFn) -> Result<()> {
    if name.is_empty() {
        return Err(ReclinkError::InvalidConfig(
            "tokenizer name cannot be empty".into(),
        ));
    }
    let mut registry = CUSTOM_TOKENIZERS.write().unwrap();
    registry.insert(name.to_string(), func);
    Ok(())
}

/// Unregister a custom tokenizer. Returns `true` if it existed.
pub fn unregister_custom_tokenizer(name: &str) -> bool {
    let mut registry = CUSTOM_TOKENIZERS.write().unwrap();
    registry.remove(name).is_some()
}

/// List all registered custom tokenizer names.
#[must_use]
pub fn list_custom_tokenizers() -> Vec<String> {
    let registry = CUSTOM_TOKENIZERS.read().unwrap();
    registry.keys().cloned().collect()
}

/// Apply a custom tokenizer by name to the given input.
///
/// # Errors
///
/// Returns an error if no tokenizer is registered under the given name.
pub fn apply_custom_tokenizer(name: &str, input: &str) -> Result<Vec<String>> {
    let registry = CUSTOM_TOKENIZERS.read().unwrap();
    if let Some(func) = registry.get(name) {
        Ok(func(input))
    } else {
        Err(ReclinkError::InvalidConfig(format!(
            "unknown custom tokenizer: `{name}`"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_use_custom_tokenizer() {
        let name = "test_tok_reg";
        let func: CustomTokenizerFn = Arc::new(|s| s.split('-').map(String::from).collect());

        register_custom_tokenizer(name, func).unwrap();

        let result = apply_custom_tokenizer(name, "hello-world-foo").unwrap();
        assert_eq!(result, vec!["hello", "world", "foo"]);

        assert!(list_custom_tokenizers().contains(&name.to_string()));
        assert!(unregister_custom_tokenizer(name));
        assert!(!unregister_custom_tokenizer(name));
        assert!(apply_custom_tokenizer(name, "test").is_err());
    }

    #[test]
    fn empty_name_rejected() {
        let func: CustomTokenizerFn = Arc::new(|s| vec![s.to_string()]);
        assert!(register_custom_tokenizer("", func).is_err());
    }
}
