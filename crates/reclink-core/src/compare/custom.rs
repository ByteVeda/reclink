//! Custom user-defined field comparator via a global registry.

use std::sync::{Arc, LazyLock, RwLock};

use ahash::AHashMap;

use crate::compare::FieldComparator;
use crate::error::{ReclinkError, Result};
use crate::record::FieldValue;

/// A custom comparator function: `(a, b) -> similarity in [0, 1]`.
pub type CustomComparatorFn = Arc<dyn Fn(&str, &str) -> f64 + Send + Sync>;

/// Global registry for custom user-defined comparators.
static CUSTOM_COMPARATORS: LazyLock<RwLock<AHashMap<String, CustomComparatorFn>>> =
    LazyLock::new(|| RwLock::new(AHashMap::new()));

/// Register a custom comparator function under the given name.
///
/// # Errors
///
/// Returns an error if the name is empty.
pub fn register_custom_comparator(name: &str, func: CustomComparatorFn) -> Result<()> {
    if name.is_empty() {
        return Err(ReclinkError::InvalidConfig(
            "comparator name cannot be empty".into(),
        ));
    }
    let mut registry = CUSTOM_COMPARATORS.write().unwrap();
    registry.insert(name.to_string(), func);
    Ok(())
}

/// Unregister a custom comparator. Returns `true` if it existed.
pub fn unregister_custom_comparator(name: &str) -> bool {
    let mut registry = CUSTOM_COMPARATORS.write().unwrap();
    registry.remove(name).is_some()
}

/// List all registered custom comparator names.
#[must_use]
pub fn list_custom_comparators() -> Vec<String> {
    let registry = CUSTOM_COMPARATORS.read().unwrap();
    registry.keys().cloned().collect()
}

/// A custom field comparator that delegates to a registered function.
pub struct CustomComparator {
    field: String,
    name: String,
    func: CustomComparatorFn,
}

/// Look up and instantiate a custom comparator by name.
///
/// # Errors
///
/// Returns an error if no comparator is registered under the given name.
pub fn custom_comparator_from_name(field: &str, name: &str) -> Result<CustomComparator> {
    let registry = CUSTOM_COMPARATORS.read().unwrap();
    if let Some(func) = registry.get(name) {
        Ok(CustomComparator {
            field: field.to_string(),
            name: name.to_string(),
            func: Arc::clone(func),
        })
    } else {
        Err(ReclinkError::InvalidConfig(format!(
            "unknown custom comparator: `{name}`"
        )))
    }
}

impl FieldComparator for CustomComparator {
    fn field_name(&self) -> &str {
        &self.field
    }

    fn compare(&self, a: &FieldValue, b: &FieldValue) -> f64 {
        if a.is_null() || b.is_null() {
            return 0.0;
        }
        (self.func)(&a.to_string(), &b.to_string())
    }
}

impl std::fmt::Debug for CustomComparator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomComparator")
            .field("field", &self.field)
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_use_custom_comparator() {
        let name = "test_cmp_reg";
        let func: CustomComparatorFn = Arc::new(|a, b| if a == b { 1.0 } else { 0.0 });

        register_custom_comparator(name, func).unwrap();

        let cmp = custom_comparator_from_name("name", name).unwrap();
        assert_eq!(cmp.field_name(), "name");

        let a = FieldValue::Text("hello".into());
        let b = FieldValue::Text("hello".into());
        assert!((cmp.compare(&a, &b) - 1.0).abs() < f64::EPSILON);

        let c = FieldValue::Text("world".into());
        assert!((cmp.compare(&a, &c) - 0.0).abs() < f64::EPSILON);

        // Null handling
        assert!((cmp.compare(&FieldValue::Null, &a) - 0.0).abs() < f64::EPSILON);

        assert!(list_custom_comparators().contains(&name.to_string()));
        assert!(unregister_custom_comparator(name));
        assert!(!unregister_custom_comparator(name));
        assert!(custom_comparator_from_name("name", name).is_err());
    }
}
