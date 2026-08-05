//! Custom user-defined classifier via a global registry.

use std::sync::{Arc, LazyLock, RwLock};

use ahash::AHashMap;

use crate::classify::Classifier;
use crate::error::{ReclinkError, Result};
use crate::record::{ClassifiedPair, ComparisonVector, MatchClass};

/// A custom classifier function: `(scores) -> (aggregate_score, match_class)`.
pub type CustomClassifierFn = Arc<dyn Fn(&[f64]) -> (f64, MatchClass) + Send + Sync>;

/// Global registry for custom user-defined classifiers.
static CUSTOM_CLASSIFIERS: LazyLock<RwLock<AHashMap<String, CustomClassifierFn>>> =
    LazyLock::new(|| RwLock::new(AHashMap::new()));

/// Register a custom classifier function under the given name.
///
/// # Errors
///
/// Returns an error if the name is empty.
pub fn register_custom_classifier(name: &str, func: CustomClassifierFn) -> Result<()> {
    if name.is_empty() {
        return Err(ReclinkError::InvalidConfig(
            "classifier name cannot be empty".into(),
        ));
    }
    let mut registry = CUSTOM_CLASSIFIERS.write().unwrap();
    registry.insert(name.to_string(), func);
    Ok(())
}

/// Unregister a custom classifier. Returns `true` if it existed.
pub fn unregister_custom_classifier(name: &str) -> bool {
    let mut registry = CUSTOM_CLASSIFIERS.write().unwrap();
    registry.remove(name).is_some()
}

/// List all registered custom classifier names.
#[must_use]
pub fn list_custom_classifiers() -> Vec<String> {
    let registry = CUSTOM_CLASSIFIERS.read().unwrap();
    registry.keys().cloned().collect()
}

/// A custom classifier that delegates to a registered function.
pub struct CustomClassifier {
    name: String,
    func: CustomClassifierFn,
}

/// Look up and instantiate a custom classifier by name.
///
/// # Errors
///
/// Returns an error if no classifier is registered under the given name.
pub fn custom_classifier_from_name(name: &str) -> Result<CustomClassifier> {
    let registry = CUSTOM_CLASSIFIERS.read().unwrap();
    if let Some(func) = registry.get(name) {
        Ok(CustomClassifier {
            name: name.to_string(),
            func: Arc::clone(func),
        })
    } else {
        Err(ReclinkError::InvalidConfig(format!(
            "unknown custom classifier: `{name}`"
        )))
    }
}

impl Classifier for CustomClassifier {
    fn classify(&self, vector: &ComparisonVector) -> ClassifiedPair {
        let (aggregate_score, class) = (self.func)(&vector.scores);
        ClassifiedPair {
            pair: vector.pair,
            scores: vector.scores.clone(),
            aggregate_score,
            class,
        }
    }
}

impl std::fmt::Debug for CustomClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomClassifier")
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::CandidatePair;

    #[test]
    fn register_and_use_custom_classifier() {
        let name = "test_cls_reg";
        let func: CustomClassifierFn = Arc::new(|scores| {
            let avg = scores.iter().sum::<f64>() / scores.len() as f64;
            if avg >= 0.8 {
                (avg, MatchClass::Match)
            } else {
                (avg, MatchClass::NonMatch)
            }
        });

        register_custom_classifier(name, func).unwrap();

        let classifier = custom_classifier_from_name(name).unwrap();
        let vector = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.9, 0.8],
        };
        let result = classifier.classify(&vector);
        assert_eq!(result.class, MatchClass::Match);
        assert!((result.aggregate_score - 0.85).abs() < f64::EPSILON);

        let vector2 = ComparisonVector {
            pair: CandidatePair { left: 0, right: 1 },
            scores: vec![0.3, 0.2],
        };
        let result2 = classifier.classify(&vector2);
        assert_eq!(result2.class, MatchClass::NonMatch);

        assert!(list_custom_classifiers().contains(&name.to_string()));
        assert!(unregister_custom_classifier(name));
        assert!(!unregister_custom_classifier(name));
        assert!(custom_classifier_from_name(name).is_err());
    }
}
