//! String field comparator.

use crate::compare::FieldComparator;
use crate::metrics::Metric;
use crate::record::FieldValue;

/// Compares text fields using any string similarity metric.
#[derive(Debug, Clone)]
pub struct StringComparator {
    field: String,
    metric: Metric,
}

impl StringComparator {
    /// Creates a new string comparator for the given field and metric.
    #[must_use]
    pub fn new(field: impl Into<String>, metric: Metric) -> Self {
        Self {
            field: field.into(),
            metric,
        }
    }
}

impl FieldComparator for StringComparator {
    fn field_name(&self) -> &str {
        &self.field
    }

    fn estimated_cost(&self) -> u32 {
        match &self.metric {
            Metric::Hamming(_) => 10,
            Metric::Jaro(_) => 20,
            Metric::JaroWinkler(_) => 20,
            Metric::Cosine(_) => 30,
            Metric::Jaccard(_) => 30,
            Metric::SorensenDice(_) => 30,
            Metric::Levenshtein(_) => 50,
            Metric::DamerauLevenshtein(_) => 50,
            Metric::WeightedLevenshtein(_) => 60,
            Metric::TokenSort(_) => 70,
            Metric::TokenSet(_) => 80,
            Metric::PartialRatio(_) => 80,
            Metric::Lcs(_) => 50,
            Metric::LongestCommonSubstring(_) => 50,
            Metric::NgramSimilarity(_) => 40,
            Metric::SmithWaterman(_) => 150,
            Metric::PhoneticHybrid(_) => 100,
            Metric::Custom { .. } => 100,
        }
    }

    fn compare(&self, a: &FieldValue, b: &FieldValue) -> f64 {
        match (a.as_text(), b.as_text()) {
            (Some(a), Some(b)) => self.metric.similarity(a, b),
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::JaroWinkler;

    #[test]
    fn string_comparison() {
        let cmp = StringComparator::new("name", Metric::JaroWinkler(JaroWinkler::default()));
        let a = FieldValue::Text("Smith".into());
        let b = FieldValue::Text("Smyth".into());
        let score = cmp.compare(&a, &b);
        assert!(score > 0.5);
    }

    #[test]
    fn null_returns_zero() {
        let cmp = StringComparator::new("name", Metric::default());
        assert_eq!(
            cmp.compare(&FieldValue::Null, &FieldValue::Text("x".into())),
            0.0
        );
    }
}
