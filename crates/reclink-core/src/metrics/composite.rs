//! Composite scorer: weighted combination of multiple metrics.

use crate::error::{ReclinkError, Result};
use crate::metrics::{metric_from_name, Metric};

/// A weighted combination of multiple metrics.
///
/// Each component contributes `weight * metric.similarity(a, b)` to the final score.
/// Weights are normalized to sum to 1.0 at construction time.
#[derive(Debug, Clone)]
pub struct CompositeScorer {
    components: Vec<(Metric, f64)>,
}

impl CompositeScorer {
    /// Create a new `CompositeScorer` from metric-weight pairs.
    ///
    /// Weights are normalized to sum to 1.0. All weights must be positive.
    ///
    /// # Errors
    ///
    /// Returns an error if no components are provided or any weight is non-positive.
    pub fn new(components: Vec<(Metric, f64)>) -> Result<Self> {
        if components.is_empty() {
            return Err(ReclinkError::EmptyInput(
                "CompositeScorer requires at least one component".to_string(),
            ));
        }
        let total: f64 = components.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            return Err(ReclinkError::InvalidConfig(
                "weights must sum to a positive value".to_string(),
            ));
        }
        let normalized: Vec<(Metric, f64)> = components
            .into_iter()
            .map(|(m, w)| (m, w / total))
            .collect();
        Ok(Self {
            components: normalized,
        })
    }

    /// Create a preset scorer by name.
    ///
    /// Available presets:
    /// - `"name_matching"`: jaro_winkler(0.5), token_sort(0.3), phonetic_hybrid(0.2)
    /// - `"address_matching"`: token_set(0.5), jaccard(0.3), levenshtein(0.2)
    /// - `"general_purpose"`: jaro_winkler(0.4), cosine(0.4), token_sort(0.2)
    ///
    /// # Errors
    ///
    /// Returns an error if the preset name is unknown.
    pub fn preset(name: &str) -> Result<Self> {
        let components: &[(&str, f64)] = match name {
            "name_matching" => &[
                ("jaro_winkler", 0.5),
                ("token_sort", 0.3),
                ("phonetic_hybrid", 0.2),
            ],
            "address_matching" => &[("token_set", 0.5), ("jaccard", 0.3), ("levenshtein", 0.2)],
            "general_purpose" => &[("jaro_winkler", 0.4), ("cosine", 0.4), ("token_sort", 0.2)],
            _ => {
                return Err(ReclinkError::InvalidConfig(format!(
                    "unknown preset: `{name}`. Expected one of: name_matching, \
                     address_matching, general_purpose"
                )));
            }
        };
        Self::from_names(components)
    }

    /// Create from string metric names and weights.
    ///
    /// # Errors
    ///
    /// Returns an error if any metric name is unknown.
    pub fn from_names(components: &[(&str, f64)]) -> Result<Self> {
        let metrics: Vec<(Metric, f64)> = components
            .iter()
            .map(|(name, weight)| Ok((metric_from_name(name)?, *weight)))
            .collect::<Result<Vec<_>>>()?;
        Self::new(metrics)
    }

    /// Compute the weighted composite similarity score.
    #[must_use]
    pub fn similarity(&self, a: &str, b: &str) -> f64 {
        self.components
            .iter()
            .map(|(metric, weight)| metric.similarity(a, b) * weight)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_metric_identity() {
        let scorer = CompositeScorer::from_names(&[("jaro_winkler", 1.0)]).unwrap();
        let a = "hello";
        let b = "hallo";
        let expected = metric_from_name("jaro_winkler").unwrap().similarity(a, b);
        assert!((scorer.similarity(a, b) - expected).abs() < 1e-10);
    }

    #[test]
    fn uniform_weights_are_average() {
        let scorer =
            CompositeScorer::from_names(&[("jaro_winkler", 1.0), ("cosine", 1.0)]).unwrap();
        let a = "hello";
        let b = "hallo";
        let jw = metric_from_name("jaro_winkler").unwrap().similarity(a, b);
        let cos = metric_from_name("cosine").unwrap().similarity(a, b);
        let expected = (jw + cos) / 2.0;
        assert!((scorer.similarity(a, b) - expected).abs() < 1e-10);
    }

    #[test]
    fn empty_components_error() {
        let result = CompositeScorer::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn preset_name_matching() {
        let scorer = CompositeScorer::preset("name_matching").unwrap();
        assert!(scorer.similarity("John Smith", "John Smith") > 0.99);
        // Reordered names should score reasonably well (token_sort handles this)
        assert!(scorer.similarity("John Smith", "Smith John") > 0.5);
    }

    #[test]
    fn preset_address_matching() {
        let scorer = CompositeScorer::preset("address_matching").unwrap();
        assert!(scorer.similarity("123 Main Street", "123 Main Street") > 0.99);
    }

    #[test]
    fn preset_general_purpose() {
        let scorer = CompositeScorer::preset("general_purpose").unwrap();
        assert!(scorer.similarity("hello", "hello") > 0.99);
    }

    #[test]
    fn preset_unknown() {
        assert!(CompositeScorer::preset("nonexistent").is_err());
    }

    #[test]
    fn identical_strings() {
        let scorer = CompositeScorer::from_names(&[
            ("jaro_winkler", 0.4),
            ("cosine", 0.3),
            ("levenshtein", 0.3),
        ])
        .unwrap();
        assert!((scorer.similarity("hello", "hello") - 1.0).abs() < 1e-10);
    }
}
