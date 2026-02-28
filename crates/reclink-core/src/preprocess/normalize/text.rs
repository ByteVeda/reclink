//! Core text manipulation primitives.

use ahash::{AHashMap, AHashSet};
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

/// Unicode normalization forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationForm {
    /// Canonical Decomposition, followed by Canonical Composition.
    Nfc,
    /// Canonical Decomposition.
    Nfd,
    /// Compatibility Decomposition, followed by Canonical Composition.
    Nfkc,
    /// Compatibility Decomposition.
    Nfkd,
}

/// Applies Unicode normalization to a string.
#[must_use]
pub fn normalize_unicode(s: &str, form: NormalizationForm) -> String {
    match form {
        NormalizationForm::Nfc => s.nfc().collect(),
        NormalizationForm::Nfd => s.nfd().collect(),
        NormalizationForm::Nfkc => s.nfkc().collect(),
        NormalizationForm::Nfkd => s.nfkd().collect(),
    }
}

/// Folds the string to lowercase.
#[must_use]
pub fn fold_case(s: &str) -> String {
    s.to_lowercase()
}

/// Normalizes whitespace: trims and collapses internal runs to single spaces.
#[must_use]
pub fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Strips all punctuation characters (Unicode category P).
#[must_use]
pub fn strip_punctuation(s: &str) -> String {
    s.chars().filter(|c| !c.is_ascii_punctuation()).collect()
}

/// Standardizes common name abbreviations and prefixes.
///
/// Expands abbreviations like "St." → "Saint", "Dr." → "Doctor", etc.
#[must_use]
pub fn standardize_name(s: &str) -> String {
    let replacements = [
        ("st.", "saint"),
        ("st ", "saint "),
        ("dr.", "doctor"),
        ("dr ", "doctor "),
        ("mr.", "mister"),
        ("mr ", "mister "),
        ("mrs.", "missus"),
        ("mrs ", "missus "),
        ("jr.", "junior"),
        ("jr ", "junior "),
        ("sr.", "senior"),
        ("sr ", "senior "),
    ];

    let mut result = s.to_lowercase();
    for (abbr, full) in &replacements {
        result = result.replace(abbr, full);
    }
    result
}

/// Removes stop words from a string.
///
/// Words are split on whitespace, filtered against the stop word set (case-insensitive),
/// and rejoined with single spaces.
#[must_use]
pub fn remove_stop_words(s: &str, stop_words: &AHashSet<String>) -> String {
    s.split_whitespace()
        .filter(|w| !stop_words.contains(&w.to_lowercase()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Strips accents/diacritics by decomposing to NFD and removing combining marks.
///
/// Uses Unicode canonical decomposition (NFD) to separate base characters from
/// their combining marks (category Mn), then filters out the marks.
#[must_use]
pub fn strip_diacritics(s: &str) -> String {
    s.nfd().filter(|c| !is_combining_mark(*c)).collect()
}

/// Applies a regex substitution to a string.
///
/// # Errors
///
/// Returns an error if the regex pattern is invalid.
pub fn regex_replace(s: &str, pattern: &str, replacement: &str) -> crate::error::Result<String> {
    let re = regex::Regex::new(pattern)
        .map_err(|e| crate::error::ReclinkError::InvalidConfig(format!("invalid regex: {e}")))?;
    Ok(re.replace_all(s, replacement).into_owned())
}

/// Expands abbreviations in a string using a lookup table.
///
/// Each whitespace-delimited token is checked (case-insensitive) against the table.
/// Matching tokens are replaced with their expansion.
#[must_use]
pub fn expand_abbreviations(s: &str, table: &AHashMap<String, String>) -> String {
    s.split_whitespace()
        .map(|w| {
            let lower = w.to_lowercase();
            table.get(&lower).cloned().unwrap_or_else(|| w.to_string())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_normalization() {
        let nfc = normalize_unicode("café", NormalizationForm::Nfc);
        let nfkc = normalize_unicode("café", NormalizationForm::Nfkc);
        assert_eq!(nfc, nfkc);
    }

    #[test]
    fn case_folding() {
        assert_eq!(fold_case("Hello WORLD"), "hello world");
    }

    #[test]
    fn whitespace_normalization() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
    }

    #[test]
    fn punctuation_stripping() {
        assert_eq!(strip_punctuation("hello, world!"), "hello world");
    }

    #[test]
    fn name_standardization() {
        assert_eq!(standardize_name("St. Louis"), "saint louis");
        assert_eq!(standardize_name("Dr. Smith"), "doctor smith");
    }

    #[test]
    fn stop_word_removal() {
        let sw = crate::preprocess::stop_words::default_english_stop_words();
        assert_eq!(remove_stop_words("the cat and the dog", &sw), "cat dog");
        assert_eq!(remove_stop_words("hello world", &sw), "hello world");
        assert_eq!(remove_stop_words("a", &sw), "");
    }

    #[test]
    fn abbreviation_expansion() {
        let table = crate::preprocess::stop_words::default_abbreviations();
        assert_eq!(
            expand_abbreviations("123 Main St.", &table),
            "123 Main street"
        );
        assert_eq!(
            expand_abbreviations("Acme Inc.", &table),
            "Acme incorporated"
        );
    }

    #[test]
    fn diacritic_stripping() {
        assert_eq!(strip_diacritics("café"), "cafe");
        assert_eq!(strip_diacritics("naïve"), "naive");
        assert_eq!(strip_diacritics("über"), "uber");
        assert_eq!(strip_diacritics("résumé"), "resume");
        // ASCII passthrough
        assert_eq!(strip_diacritics("hello"), "hello");
        assert_eq!(strip_diacritics(""), "");
    }

    #[test]
    fn regex_replacement() {
        assert_eq!(
            regex_replace("hello 123 world", r"\d+", "").unwrap(),
            "hello  world"
        );
        assert_eq!(
            regex_replace("foo-bar-baz", r"-", " ").unwrap(),
            "foo bar baz"
        );
        // Invalid regex
        assert!(regex_replace("test", r"[invalid", "").is_err());
    }
}
