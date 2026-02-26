//! Binary phonetic field comparator.

use crate::compare::FieldComparator;
use crate::phonetic::{PhoneticAlgorithm, PhoneticEncoder};
use crate::record::FieldValue;

/// Returns 1.0 if the phonetic codes match, 0.0 otherwise.
#[derive(Debug, Clone)]
pub struct PhoneticComparator {
    field: String,
    algorithm: PhoneticAlgorithm,
}

impl PhoneticComparator {
    /// Creates a new phonetic comparator.
    #[must_use]
    pub fn new(field: impl Into<String>, algorithm: PhoneticAlgorithm) -> Self {
        Self {
            field: field.into(),
            algorithm,
        }
    }
}

impl FieldComparator for PhoneticComparator {
    fn field_name(&self) -> &str {
        &self.field
    }

    fn compare(&self, a: &FieldValue, b: &FieldValue) -> f64 {
        let a_text = match a {
            FieldValue::Text(s) => s.as_str(),
            _ => return 0.0,
        };
        let b_text = match b {
            FieldValue::Text(s) => s.as_str(),
            _ => return 0.0,
        };

        if self.algorithm.encode(a_text) == self.algorithm.encode(b_text) {
            1.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::Soundex;

    #[test]
    fn phonetic_match() {
        let cmp = PhoneticComparator::new("name", PhoneticAlgorithm::Soundex(Soundex));
        assert_eq!(
            cmp.compare(
                &FieldValue::Text("Smith".into()),
                &FieldValue::Text("Smyth".into())
            ),
            1.0
        );
    }

    #[test]
    fn phonetic_no_match() {
        let cmp = PhoneticComparator::new("name", PhoneticAlgorithm::Soundex(Soundex));
        assert_eq!(
            cmp.compare(
                &FieldValue::Text("Smith".into()),
                &FieldValue::Text("Jones".into())
            ),
            0.0
        );
    }

    #[test]
    fn null_returns_zero() {
        let cmp = PhoneticComparator::new("name", PhoneticAlgorithm::Soundex(Soundex));
        assert_eq!(
            cmp.compare(&FieldValue::Null, &FieldValue::Text("Smith".into())),
            0.0
        );
    }
}
