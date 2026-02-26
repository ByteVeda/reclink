//! Text normalization utilities.

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
}
