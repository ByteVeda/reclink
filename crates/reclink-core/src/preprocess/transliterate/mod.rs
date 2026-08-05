//! Transliteration from non-Latin scripts to Latin characters.
//!
//! Provides transliteration for Cyrillic, Greek, Arabic, Hebrew, Devanagari,
//! and Hangul scripts via static lookup tables and algorithmic decomposition.
//! Designed for preprocessing names and addresses in multilingual datasets.

mod arabic;
mod cyrillic;
mod devanagari;
mod greek;
mod hangul;
mod hebrew;

/// Target script for transliteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    /// Cyrillic → Latin (ISO 9 / BGN/PCGN inspired).
    Cyrillic,
    /// Greek → Latin (UN/ELOT 743 inspired).
    Greek,
    /// Arabic → Latin (simplified scholarly romanization).
    Arabic,
    /// Hebrew → Latin (simplified common romanization).
    Hebrew,
    /// Devanagari → Latin (IAST-style romanization).
    Devanagari,
    /// Hangul → Latin (Revised Romanization of Korean).
    Hangul,
}

/// Transliterate a string from the given script to Latin characters.
///
/// Characters not in the lookup table are passed through unchanged,
/// so mixed-script text works correctly.
#[must_use]
pub fn transliterate(s: &str, script: Script) -> String {
    match script {
        Script::Cyrillic => cyrillic::transliterate_cyrillic(s),
        Script::Greek => greek::transliterate_greek(s),
        Script::Arabic => arabic::transliterate_arabic(s),
        Script::Hebrew => hebrew::transliterate_hebrew(s),
        Script::Devanagari => devanagari::transliterate_devanagari(s),
        Script::Hangul => hangul::transliterate_hangul(s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string() {
        assert_eq!(transliterate("", Script::Cyrillic), "");
        assert_eq!(transliterate("", Script::Greek), "");
    }

    #[test]
    fn pure_latin_passthrough() {
        assert_eq!(
            transliterate("hello world", Script::Cyrillic),
            "hello world"
        );
        assert_eq!(transliterate("hello world", Script::Greek), "hello world");
    }
}
