//! Hamming distance for equal-length strings.

use crate::error::{ReclinkError, Result};
use crate::metrics::DistanceMetric;

/// Hamming distance counts the number of positions where corresponding
/// characters differ. Requires strings of equal length.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Hamming;

impl DistanceMetric for Hamming {
    fn distance(&self, a: &str, b: &str) -> Result<usize> {
        hamming_distance(a, b)
    }
}

/// Computes Hamming distance between two equal-length strings.
///
/// Returns an error if the strings have different lengths.
pub fn hamming_distance(a: &str, b: &str) -> Result<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.len() != b_chars.len() {
        return Err(ReclinkError::UnequalLength {
            a: a_chars.len(),
            b: b_chars.len(),
        });
    }

    Ok(a_chars
        .iter()
        .zip(b_chars.iter())
        .filter(|(a, b)| a != b)
        .count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical() {
        assert_eq!(hamming_distance("hello", "hello").unwrap(), 0);
    }

    #[test]
    fn empty() {
        assert_eq!(hamming_distance("", "").unwrap(), 0);
    }

    #[test]
    fn known_values() {
        assert_eq!(hamming_distance("karolin", "kathrin").unwrap(), 3);
        assert_eq!(hamming_distance("1011101", "1001001").unwrap(), 2);
    }

    #[test]
    fn unequal_length_error() {
        assert!(hamming_distance("abc", "ab").is_err());
    }

    #[test]
    fn unicode() {
        assert_eq!(hamming_distance("café", "cafè").unwrap(), 1);
    }
}
