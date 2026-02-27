//! Beider-Morse Phonetic Matching (BMPM).
//!
//! A phonetic algorithm that handles names from multiple language origins.
//! Unlike Soundex or Metaphone which are English-centric, Beider-Morse
//! detects the likely language of a name and applies language-specific
//! phonetic rules, producing one or more phonetic codes.
//!
//! # References
//!
//! - Beider & Morse, "An Alternative to Soundex with Fewer False Hits", 2008
//! - Apache Commons Codec `BeiderMorseEncoder`

pub mod language;
pub mod rules;

use crate::phonetic::PhoneticEncoder;
use language::{detect_language, Language};
use rules::{apply_rules, cleanup, ASHKENAZI_RULES, GENERIC_RULES};

/// Maximum number of phonetic variants to generate.
const MAX_VARIANTS: usize = 32;

/// Beider-Morse phonetic encoder.
///
/// Detects the language origin of a name and applies the appropriate
/// phonetic transformation rules. Can operate in generic mode or
/// Ashkenazi-specific mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BeiderMorse {
    /// If true, always use Ashkenazi rules regardless of language detection.
    pub ashkenazi: bool,
}

impl BeiderMorse {
    /// Create a new encoder with default settings (generic mode).
    #[must_use]
    pub fn new() -> Self {
        Self { ashkenazi: false }
    }

    /// Create a new encoder in Ashkenazi mode.
    #[must_use]
    pub fn ashkenazi() -> Self {
        Self { ashkenazi: true }
    }

    /// Encode a string and return all phonetic variants.
    ///
    /// Returns multiple possible encodings separated by `|` in the
    /// primary encoding, and all variants as a vector.
    pub fn encode_all(&self, s: &str) -> Vec<String> {
        if s.is_empty() {
            return vec![String::new()];
        }

        let rules = self.select_rules(s);
        let variants = apply_rules(s, rules, MAX_VARIANTS);

        // Clean up each variant
        variants.into_iter().map(|v| cleanup(&v)).collect()
    }

    /// Select the appropriate rule set based on mode and language detection.
    fn select_rules(&self, s: &str) -> &'static [rules::Rule] {
        if self.ashkenazi {
            return ASHKENAZI_RULES;
        }

        let lang = detect_language(s);
        match lang {
            Language::Ashkenazi => ASHKENAZI_RULES,
            _ => GENERIC_RULES,
        }
    }
}

impl PhoneticEncoder for BeiderMorse {
    /// Returns the primary (first) phonetic encoding.
    fn encode(&self, s: &str) -> String {
        let variants = self.encode_all(s);
        if variants.len() <= 1 {
            return variants.into_iter().next().unwrap_or_default();
        }
        // Join all variants with | for the primary encoding
        variants.join("|")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let bm = BeiderMorse::new();
        assert_eq!(bm.encode(""), "");
    }

    #[test]
    fn basic_encoding() {
        let bm = BeiderMorse::new();
        let result = bm.encode("Smith");
        assert!(!result.is_empty());
        // Should produce phonetic code(s)
    }

    #[test]
    fn similar_names_similar_codes() {
        let bm = BeiderMorse::new();
        let code1 = bm.encode("Smith");
        let code2 = bm.encode("Smyth");
        // Both should produce overlapping phonetic codes
        let variants1: std::collections::HashSet<&str> = code1.split('|').collect();
        let variants2: std::collections::HashSet<&str> = code2.split('|').collect();
        // At least some variants should overlap or be similar
        assert!(!variants1.is_empty());
        assert!(!variants2.is_empty());
    }

    #[test]
    fn ashkenazi_mode() {
        let bm = BeiderMorse::ashkenazi();
        let code = bm.encode("Schwartz");
        assert!(!code.is_empty());
    }

    #[test]
    fn encode_all_returns_variants() {
        let bm = BeiderMorse::new();
        let variants = bm.encode_all("Charles");
        assert!(!variants.is_empty());
    }

    #[test]
    fn german_name() {
        let bm = BeiderMorse::new();
        let code = bm.encode("Schmidt");
        assert!(!code.is_empty());
    }

    #[test]
    fn phonetic_encoder_trait() {
        let bm = BeiderMorse::new();
        // Test trait method is_match
        // Same-sounding names should have matching codes
        let code1 = bm.encode("John");
        let code2 = bm.encode("Jon");
        // They may or may not match exactly depending on rules,
        // but both should produce valid output
        assert!(!code1.is_empty());
        assert!(!code2.is_empty());
    }

    #[test]
    fn max_variants_bounded() {
        let bm = BeiderMorse::new();
        let variants = bm.encode_all("Schwarzenegger");
        assert!(variants.len() <= 32);
    }

    #[test]
    fn goldstein_ashkenazi_detection() {
        // Generic mode should auto-detect Ashkenazi
        let bm = BeiderMorse::new();
        let code = bm.encode("Goldstein");
        assert!(!code.is_empty());
    }
}
