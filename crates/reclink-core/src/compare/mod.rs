//! Field comparators for computing similarity between record fields.

pub mod custom;
mod date;
mod exact;
mod numeric;
mod phonetic;
mod string;

pub use custom::*;
pub use date::DateComparator;
pub use exact::ExactComparator;
pub use numeric::NumericComparator;
pub use phonetic::PhoneticComparator;
pub use string::StringComparator;

use crate::record::FieldValue;

/// Trait for field-level comparators.
pub trait FieldComparator: Send + Sync {
    /// Returns the field name this comparator operates on.
    fn field_name(&self) -> &str;

    /// Computes similarity between two field values, returning a score in \[0, 1\].
    fn compare(&self, a: &FieldValue, b: &FieldValue) -> f64;
}
