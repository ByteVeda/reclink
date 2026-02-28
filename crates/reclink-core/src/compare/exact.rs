//! Exact match field comparator.

use crate::compare::FieldComparator;
use crate::record::FieldValue;

/// Returns 1.0 for exact field equality, 0.0 otherwise.
#[derive(Debug, Clone)]
pub struct ExactComparator {
    field: String,
}

impl ExactComparator {
    /// Creates a new exact comparator.
    #[must_use]
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl FieldComparator for ExactComparator {
    fn field_name(&self) -> &str {
        &self.field
    }

    fn estimated_cost(&self) -> u32 {
        1
    }

    fn compare(&self, a: &FieldValue, b: &FieldValue) -> f64 {
        if a == b && !a.is_null() {
            1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let cmp = ExactComparator::new("city");
        assert_eq!(
            cmp.compare(
                &FieldValue::Text("NYC".into()),
                &FieldValue::Text("NYC".into())
            ),
            1.0
        );
    }

    #[test]
    fn no_match() {
        let cmp = ExactComparator::new("city");
        assert_eq!(
            cmp.compare(
                &FieldValue::Text("NYC".into()),
                &FieldValue::Text("LA".into())
            ),
            0.0
        );
    }

    #[test]
    fn null_no_match() {
        let cmp = ExactComparator::new("city");
        assert_eq!(cmp.compare(&FieldValue::Null, &FieldValue::Null), 0.0);
    }
}
