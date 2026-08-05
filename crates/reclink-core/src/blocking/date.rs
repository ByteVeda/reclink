//! Date blocking by truncation resolution.

use ahash::AHashMap;

use crate::blocking::BlockingStrategy;
use crate::record::{CandidatePair, RecordBatch};

/// Resolution for date blocking truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateResolution {
    /// Block by year (e.g., "1990").
    Year,
    /// Block by year and month (e.g., "1990-01").
    Month,
    /// Block by full date (e.g., "1990-01-15").
    Day,
}

/// Blocks records by truncating a date field to a given resolution.
#[derive(Debug, Clone)]
pub struct DateBlocking {
    /// The field name to block on.
    pub field: String,
    /// The truncation resolution.
    pub resolution: DateResolution,
}

impl DateBlocking {
    /// Creates a new date blocker.
    #[must_use]
    pub fn new(field: impl Into<String>, resolution: DateResolution) -> Self {
        Self {
            field: field.into(),
            resolution,
        }
    }

    fn get_key(&self, record: &crate::record::Record) -> Option<String> {
        let date_str = match record.get(&self.field)? {
            crate::record::FieldValue::Date(d) => d.as_str(),
            crate::record::FieldValue::Text(t) => t.as_str(),
            _ => return None,
        };

        let parts: Vec<&str> = date_str.split('-').collect();
        if parts.is_empty() {
            return None;
        }

        let key = match self.resolution {
            DateResolution::Year => parts.first()?.to_string(),
            DateResolution::Month => {
                if parts.len() >= 2 {
                    format!("{}-{}", parts[0], parts[1])
                } else {
                    parts[0].to_string()
                }
            }
            DateResolution::Day => date_str.to_string(),
        };
        Some(key)
    }
}

impl BlockingStrategy for DateBlocking {
    fn block_dedup(&self, records: &RecordBatch) -> Vec<CandidatePair> {
        let mut blocks: AHashMap<String, Vec<usize>> = AHashMap::new();

        for (i, record) in records.records.iter().enumerate() {
            if let Some(key) = self.get_key(record) {
                blocks.entry(key).or_default().push(i);
            }
        }

        let mut pairs = Vec::new();
        for indices in blocks.values() {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    pairs.push(CandidatePair {
                        left: indices[i],
                        right: indices[j],
                    });
                }
            }
        }
        pairs
    }

    fn block_link(&self, left: &RecordBatch, right: &RecordBatch) -> Vec<CandidatePair> {
        let mut right_index: AHashMap<String, Vec<usize>> = AHashMap::new();
        for (i, record) in right.records.iter().enumerate() {
            if let Some(key) = self.get_key(record) {
                right_index.entry(key).or_default().push(i);
            }
        }

        let mut pairs = Vec::new();
        for (i, record) in left.records.iter().enumerate() {
            if let Some(key) = self.get_key(record) {
                if let Some(matches) = right_index.get(&key) {
                    for &j in matches {
                        pairs.push(CandidatePair { left: i, right: j });
                    }
                }
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{FieldValue, Record, RecordBatch};

    fn make_date_batch(dates: &[&str]) -> RecordBatch {
        let records: Vec<Record> = dates
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                Record::new(format!("{}", i + 1)).with_field("dob", FieldValue::Date(d.to_string()))
            })
            .collect();
        RecordBatch::new(vec!["dob".to_string()], records)
    }

    #[test]
    fn date_block_year() {
        let batch = make_date_batch(&["1990-01-15", "1990-06-20", "2000-01-01"]);
        let blocker = DateBlocking::new("dob", DateResolution::Year);
        let pairs = blocker.block_dedup(&batch);
        // Records 0 and 1 share year 1990
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 1);
    }

    #[test]
    fn date_block_month() {
        let batch = make_date_batch(&["1990-01-15", "1990-01-20", "1990-06-01"]);
        let blocker = DateBlocking::new("dob", DateResolution::Month);
        let pairs = blocker.block_dedup(&batch);
        // Records 0 and 1 share 1990-01
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 1);
    }

    #[test]
    fn date_block_different() {
        let batch = make_date_batch(&["1990-01-15", "2000-06-20"]);
        let blocker = DateBlocking::new("dob", DateResolution::Year);
        let pairs = blocker.block_dedup(&batch);
        assert!(pairs.is_empty());
    }
}
