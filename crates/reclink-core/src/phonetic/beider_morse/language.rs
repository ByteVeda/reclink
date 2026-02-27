//! Language detection for Beider-Morse phonetic matching.
//!
//! Detects the likely language/origin of a name based on character patterns
//! and letter combinations, then selects the appropriate phonetic rule set.

/// Supported language/origin categories for Beider-Morse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// Generic (no specific language detected).
    Generic,
    /// English names.
    English,
    /// German names.
    German,
    /// French names.
    French,
    /// Spanish names.
    Spanish,
    /// Italian names.
    Italian,
    /// Portuguese names.
    Portuguese,
    /// Polish names.
    Polish,
    /// Russian/Slavic names (transliterated).
    Russian,
    /// Ashkenazi Hebrew-origin names.
    Ashkenazi,
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Language::Generic => "generic",
            Language::English => "english",
            Language::German => "german",
            Language::French => "french",
            Language::Spanish => "spanish",
            Language::Italian => "italian",
            Language::Portuguese => "portuguese",
            Language::Polish => "polish",
            Language::Russian => "russian",
            Language::Ashkenazi => "ashkenazi",
        };
        write!(f, "{name}")
    }
}

/// Detect the most likely language for a given name.
///
/// Uses character patterns and common letter combinations to guess the
/// language origin. Falls back to `Language::Generic` when uncertain.
pub fn detect_language(s: &str) -> Language {
    let lower = s.to_lowercase();

    // Check for Ashkenazi indicators
    if has_ashkenazi_pattern(&lower) {
        return Language::Ashkenazi;
    }

    // Check for Slavic/Polish patterns
    if has_polish_pattern(&lower) {
        return Language::Polish;
    }

    // Check for Russian (transliterated) patterns
    if has_russian_pattern(&lower) {
        return Language::Russian;
    }

    // Check for German patterns
    if has_german_pattern(&lower) {
        return Language::German;
    }

    // Check for French patterns
    if has_french_pattern(&lower) {
        return Language::French;
    }

    // Check for Italian patterns
    if has_italian_pattern(&lower) {
        return Language::Italian;
    }

    // Check for Spanish patterns
    if has_spanish_pattern(&lower) {
        return Language::Spanish;
    }

    // Check for Portuguese patterns
    if has_portuguese_pattern(&lower) {
        return Language::Portuguese;
    }

    // Check for English patterns
    if has_english_pattern(&lower) {
        return Language::English;
    }

    Language::Generic
}

fn has_ashkenazi_pattern(s: &str) -> bool {
    // Common Ashkenazi name patterns
    s.contains("tz") && s.contains("man")
        || s.ends_with("witz")
        || s.ends_with("vitz")
        || s.ends_with("stein")
        || s.ends_with("berg")
        || s.ends_with("feld")
        || s.ends_with("baum")
        || s.ends_with("blatt")
        || s.starts_with("gold")
        || s.starts_with("silver")
        || s.starts_with("rosen")
}

fn has_polish_pattern(s: &str) -> bool {
    s.contains("sz") && !s.contains("sch")
        || s.contains("cz") && !s.contains("sch")
        || s.ends_with("ski")
        || s.ends_with("ska")
        || s.ends_with("icz")
        || s.ends_with("iak")
}

fn has_russian_pattern(s: &str) -> bool {
    s.ends_with("ov")
        || s.ends_with("ev")
        || s.ends_with("ova")
        || s.ends_with("eva")
        || s.ends_with("enko")
        || s.ends_with("enko")
        || s.contains("kh")
        || s.contains("zh")
}

fn has_german_pattern(s: &str) -> bool {
    s.contains("sch")
        || s.contains("tsch")
        || s.ends_with("burg")
        || s.ends_with("dorf")
        || s.ends_with("mann")
        || s.ends_with("meier")
        || s.ends_with("mayer")
        || s.contains("ei") && s.contains("er")
}

fn has_french_pattern(s: &str) -> bool {
    s.ends_with("eau")
        || s.ends_with("eux")
        || s.ends_with("ault")
        || s.ends_with("ois")
        || s.ends_with("ais")
        || s.contains("ou") && s.ends_with("er")
        || s.starts_with("de ")
        || s.starts_with("le ")
}

fn has_italian_pattern(s: &str) -> bool {
    s.ends_with("ini")
        || s.ends_with("elli")
        || s.ends_with("etti")
        || s.ends_with("ucci")
        || s.ends_with("acci")
        || s.ends_with("otti")
        || s.ends_with("ano")
        || s.ends_with("ino") && s.len() > 4
}

fn has_spanish_pattern(s: &str) -> bool {
    s.ends_with("ez")
        || s.ends_with("az")
        || s.ends_with("ero")
        || s.contains("ll") && (s.ends_with("a") || s.ends_with("o"))
}

fn has_portuguese_pattern(s: &str) -> bool {
    s.ends_with("ões")
        || s.ends_with("ães")
        || s.ends_with("eira")
        || s.ends_with("eiro")
        || s.contains("lh")
        || s.contains("nh") && !s.contains("nho")
}

fn has_english_pattern(s: &str) -> bool {
    s.ends_with("tion")
        || s.ends_with("ght")
        || s.ends_with("ley")
        || s.ends_with("ton")
        || s.contains("th") && s.contains("er")
        || s.ends_with("son")
        || s.ends_with("ham")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_ashkenazi() {
        assert_eq!(detect_language("Goldstein"), Language::Ashkenazi);
        assert_eq!(detect_language("Rosenberg"), Language::Ashkenazi);
        assert_eq!(detect_language("Horowitz"), Language::Ashkenazi);
    }

    #[test]
    fn detect_german() {
        assert_eq!(detect_language("Schmidt"), Language::German);
        assert_eq!(detect_language("Schwarzenegger"), Language::German);
    }

    #[test]
    fn detect_french() {
        assert_eq!(detect_language("Rousseau"), Language::French);
    }

    #[test]
    fn detect_polish() {
        assert_eq!(detect_language("Kowalski"), Language::Polish);
    }

    #[test]
    fn detect_russian() {
        assert_eq!(detect_language("Petrov"), Language::Russian);
        assert_eq!(detect_language("Ivanova"), Language::Russian);
    }

    #[test]
    fn detect_italian() {
        assert_eq!(detect_language("Rossini"), Language::Italian);
        assert_eq!(detect_language("Moretti"), Language::Italian);
    }

    #[test]
    fn detect_spanish() {
        assert_eq!(detect_language("Rodriguez"), Language::Spanish);
        assert_eq!(detect_language("Gonzalez"), Language::Spanish);
    }

    #[test]
    fn detect_english() {
        assert_eq!(detect_language("Washington"), Language::English);
        assert_eq!(detect_language("Thompson"), Language::English);
    }

    #[test]
    fn detect_generic_fallback() {
        assert_eq!(detect_language("Kim"), Language::Generic);
    }
}
