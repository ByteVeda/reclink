//! Date field comparator.

use crate::compare::FieldComparator;
use crate::record::FieldValue;

/// Compares date fields by checking exact string equality or partial matches.
#[derive(Debug, Clone)]
pub struct DateComparator {
    field: String,
}

impl DateComparator {
    /// Creates a new date comparator.
    #[must_use]
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl FieldComparator for DateComparator {
    fn field_name(&self) -> &str {
        &self.field
    }

    fn compare(&self, a: &FieldValue, b: &FieldValue) -> f64 {
        let a_str = match a {
            FieldValue::Date(d) => d.as_str(),
            FieldValue::Text(t) => t.as_str(),
            _ => return 0.0,
        };
        let b_str = match b {
            FieldValue::Date(d) => d.as_str(),
            FieldValue::Text(t) => t.as_str(),
            _ => return 0.0,
        };

        if a_str == b_str {
            return 1.0;
        }

        // Partial date matching: compare year, month, day components
        let a_parts: Vec<&str> = a_str.split('-').collect();
        let b_parts: Vec<&str> = b_str.split('-').collect();

        if a_parts.len() != 3 || b_parts.len() != 3 {
            return 0.0;
        }

        let mut score = 0.0;
        let weights = [0.4, 0.3, 0.3]; // year, month, day

        for (i, weight) in weights.iter().enumerate() {
            if a_parts[i] == b_parts[i] {
                score += weight;
            }
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_date() {
        let cmp = DateComparator::new("dob");
        assert_eq!(
            cmp.compare(
                &FieldValue::Date("1990-01-15".into()),
                &FieldValue::Date("1990-01-15".into())
            ),
            1.0
        );
    }

    #[test]
    fn partial_date() {
        let cmp = DateComparator::new("dob");
        let score = cmp.compare(
            &FieldValue::Date("1990-01-15".into()),
            &FieldValue::Date("1990-01-20".into()),
        );
        assert!((score - 0.7).abs() < 1e-10); // year + month match
    }

    #[test]
    fn different_date() {
        let cmp = DateComparator::new("dob");
        assert_eq!(
            cmp.compare(
                &FieldValue::Date("1990-01-15".into()),
                &FieldValue::Date("2000-06-20".into())
            ),
            0.0
        );
    }
}
