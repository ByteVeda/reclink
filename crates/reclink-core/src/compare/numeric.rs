//! Numeric field comparator.

use crate::compare::FieldComparator;
use crate::record::FieldValue;

/// Compares numeric fields using absolute or relative difference.
#[derive(Debug, Clone)]
pub struct NumericComparator {
    field: String,
    /// Maximum difference for a score of 0. Differences beyond this are 0 similarity.
    max_diff: f64,
}

impl NumericComparator {
    /// Creates a new numeric comparator with a maximum difference threshold.
    #[must_use]
    pub fn new(field: impl Into<String>, max_diff: f64) -> Self {
        Self {
            field: field.into(),
            max_diff,
        }
    }
}

impl FieldComparator for NumericComparator {
    fn field_name(&self) -> &str {
        &self.field
    }

    fn estimated_cost(&self) -> u32 {
        2
    }

    fn selectivity_hint(&self) -> f64 {
        3.0
    }

    fn compare(&self, a: &FieldValue, b: &FieldValue) -> f64 {
        let a_val = match a {
            FieldValue::Integer(i) => *i as f64,
            FieldValue::Float(f) => *f,
            _ => return 0.0,
        };
        let b_val = match b {
            FieldValue::Integer(i) => *i as f64,
            FieldValue::Float(f) => *f,
            _ => return 0.0,
        };

        let diff = (a_val - b_val).abs();
        if self.max_diff <= 0.0 {
            return if diff == 0.0 { 1.0 } else { 0.0 };
        }

        (1.0 - diff / self.max_diff).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let cmp = NumericComparator::new("age", 10.0);
        assert_eq!(
            cmp.compare(&FieldValue::Integer(25), &FieldValue::Integer(25)),
            1.0
        );
    }

    #[test]
    fn partial_match() {
        let cmp = NumericComparator::new("age", 10.0);
        let score = cmp.compare(&FieldValue::Integer(25), &FieldValue::Integer(30));
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn beyond_threshold() {
        let cmp = NumericComparator::new("age", 10.0);
        assert_eq!(
            cmp.compare(&FieldValue::Integer(0), &FieldValue::Integer(20)),
            0.0
        );
    }
}
