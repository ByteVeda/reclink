//! Phonetic encoding algorithms for name matching.
//!
//! Provides Soundex, Metaphone, Double Metaphone, NYSIIS, Caverphone,
//! and Cologne Phonetic encoders through a common [`PhoneticEncoder`] trait.

pub mod beider_morse;
mod caverphone;
mod cologne;
mod double_metaphone;
mod metaphone;
mod nysiis;
mod soundex;

pub use beider_morse::language::{detect_language, Language};
pub use beider_morse::BeiderMorse;
pub use caverphone::Caverphone;
pub use cologne::ColognePhonetic;
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum PhoneticAlgorithm {
    /// Soundex encoding.
    Soundex(Soundex),
    /// Metaphone encoding.
    Metaphone(Metaphone),
    /// Double Metaphone encoding.
    DoubleMetaphone(DoubleMetaphone),
    /// NYSIIS encoding.
    Nysiis(Nysiis),
    /// Caverphone 2 encoding.
    Caverphone(Caverphone),
    /// Cologne Phonetic (Kolner Phonetik) encoding.
    ColognePhonetic(ColognePhonetic),
    /// Beider-Morse Phonetic Matching.
    BeiderMorse(BeiderMorse),
}

impl PhoneticEncoder for PhoneticAlgorithm {
    fn encode(&self, s: &str) -> String {
        match self {
            PhoneticAlgorithm::Soundex(e) => e.encode(s),
            PhoneticAlgorithm::Metaphone(e) => e.encode(s),
            PhoneticAlgorithm::DoubleMetaphone(e) => e.encode(s),
            PhoneticAlgorithm::Nysiis(e) => e.encode(s),
            PhoneticAlgorithm::Caverphone(e) => e.encode(s),
            PhoneticAlgorithm::ColognePhonetic(e) => e.encode(s),
            PhoneticAlgorithm::BeiderMorse(e) => e.encode(s),
        }
    }
}
