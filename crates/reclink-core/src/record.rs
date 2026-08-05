//! Core record types for the record linkage pipeline.

use ahash::AHashMap;
use std::fmt;

/// A value stored in a record field.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// A text value.
    Text(String),
    /// An integer value.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
    /// A date string (ISO 8601 format).
    Date(String),
    /// A null/missing value.
    Null,
}

impl fmt::Display for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldValue::Text(s) => write!(f, "{s}"),
            FieldValue::Integer(i) => write!(f, "{i}"),
            FieldValue::Float(v) => write!(f, "{v}"),
            FieldValue::Date(d) => write!(f, "{d}"),
            FieldValue::Null => write!(f, "NULL"),
        }
    }
}

impl FieldValue {
    /// Returns the text content if this is a `Text` variant.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            FieldValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the integer content if this is an `Integer` variant.
    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            FieldValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the float content if this is a `Float` variant.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            FieldValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the date string if this is a `Date` variant.
    #[must_use]
    pub fn as_date(&self) -> Option<&str> {
        match self {
            FieldValue::Date(d) => Some(d),
            _ => None,
        }
    }

    /// Extracts a numeric value as f64 from Integer, Float, or parseable Text.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FieldValue::Integer(i) => Some(*i as f64),
            FieldValue::Float(f) => Some(*f),
            FieldValue::Text(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Returns true if this is a `Null` variant.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, FieldValue::Null)
    }
}

/// A single record with an identifier and a set of named fields.
#[derive(Debug, Clone)]
pub struct Record {
    /// Unique identifier for this record.
    pub id: String,
    /// Field name to value mapping.
    pub fields: AHashMap<String, FieldValue>,
}

impl Record {
    /// Creates a new record with the given id and empty fields.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: AHashMap::new(),
        }
    }

    /// Inserts a field value and returns `self` for chaining.
    pub fn with_field(mut self, name: impl Into<String>, value: FieldValue) -> Self {
        self.fields.insert(name.into(), value);
        self
    }

    /// Gets a field value by name.
    #[must_use]
    pub fn get(&self, field: &str) -> Option<&FieldValue> {
        self.fields.get(field)
    }

    /// Gets a text field value by name, returning `None` if missing or wrong type.
    #[must_use]
    pub fn get_text(&self, field: &str) -> Option<&str> {
        self.fields.get(field).and_then(|v| v.as_text())
    }
}

/// A batch of records sharing a common schema.
#[derive(Debug, Clone)]
pub struct RecordBatch {
    /// Ordered list of field names (the schema).
    pub field_names: Vec<String>,
    /// The records in this batch.
    pub records: Vec<Record>,
}

impl RecordBatch {
    /// Creates a new batch with the given schema and records.
    #[must_use]
    pub fn new(field_names: Vec<String>, records: Vec<Record>) -> Self {
        Self {
            field_names,
            records,
        }
    }

    /// Returns the number of records in this batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if the batch contains no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// A pair of record indices that are candidates for matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CandidatePair {
    /// Index of the first record (in source A or the dedup dataset).
    pub left: usize,
    /// Index of the second record (in source B or the dedup dataset).
    pub right: usize,
}

/// A vector of comparison scores for a candidate pair (one score per compared field).
#[derive(Debug, Clone)]
pub struct ComparisonVector {
    /// The candidate pair.
    pub pair: CandidatePair,
    /// Similarity scores for each compared field, in order.
    pub scores: Vec<f64>,
}

/// The result of classifying a comparison vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatchClass {
    /// The pair is a match.
    Match,
    /// The pair is not a match.
    NonMatch,
    /// The pair is in the uncertain zone (Fellegi-Sunter).
    Possible,
}

/// A candidate pair with its classification result.
#[derive(Debug, Clone)]
pub struct ClassifiedPair {
    /// The candidate pair.
    pub pair: CandidatePair,
    /// The comparison scores.
    pub scores: Vec<f64>,
    /// The aggregate score (e.g., weighted sum).
    pub aggregate_score: f64,
    /// The classification result.
    pub class: MatchClass,
}
