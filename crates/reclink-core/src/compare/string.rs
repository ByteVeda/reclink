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
