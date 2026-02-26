//! Phonetic encoding algorithms for name matching.
//!
//! Provides Soundex, Metaphone, Double Metaphone, and NYSIIS encoders
//! through a common [`PhoneticEncoder`] trait.

mod double_metaphone;
mod metaphone;
mod nysiis;
mod soundex;

pub use double_metaphone::DoubleMetaphone;
pub use metaphone::Metaphone;
pub use nysiis::Nysiis;
pub use soundex::Soundex;

/// Trait for phonetic encoding algorithms.
pub trait PhoneticEncoder {
    /// Encodes a string into its phonetic representation.
    fn encode(&self, s: &str) -> String;

    /// Encodes each string in a slice.
    fn encode_all(&self, strings: &[&str]) -> Vec<String> {
        strings.iter().map(|s| self.encode(s)).collect()
    }

    /// Returns true if two strings have the same phonetic encoding.
    fn is_match(&self, a: &str, b: &str) -> bool {
        self.encode(a) == self.encode(b)
    }
}

/// Enum dispatch for phonetic encoders.
#[derive(Debug, Clone)]
pub enum PhoneticAlgorithm {
    /// Soundex encoding.
    Soundex(Soundex),
    /// Metaphone encoding.
    Metaphone(Metaphone),
    /// Double Metaphone encoding.
    DoubleMetaphone(DoubleMetaphone),
    /// NYSIIS encoding.
    Nysiis(Nysiis),
}

impl PhoneticEncoder for PhoneticAlgorithm {
    fn encode(&self, s: &str) -> String {
        match self {
            PhoneticAlgorithm::Soundex(e) => e.encode(s),
            PhoneticAlgorithm::Metaphone(e) => e.encode(s),
            PhoneticAlgorithm::DoubleMetaphone(e) => e.encode(s),
            PhoneticAlgorithm::Nysiis(e) => e.encode(s),
        }
    }
}
