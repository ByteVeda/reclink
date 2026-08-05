//! Script-specific normalization for Arabic, Hebrew, and bidirectional text.

/// Strips Arabic diacritics (harakat, tatweel, superscript alef).
#[must_use]
pub fn strip_arabic_diacritics(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !matches!(
                c,
                '\u{0640}'            // Tatweel
                | '\u{064B}'
                    ..='\u{065F}' // Harakat (fathatan through hamza below)
                | '\u{0670}' // Superscript Alef
            )
        })
        .collect()
}

/// Strips Hebrew diacritics (niqqud and cantillation marks).
#[must_use]
pub fn strip_hebrew_diacritics(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !matches!(c,
                '\u{05B0}'..='\u{05BD}' // Niqqud (sheva through meteg)
                | '\u{05BF}'            // Rafe
                | '\u{05C1}'..='\u{05C2}' // Shin/Sin dot
                | '\u{05C4}'..='\u{05C5}' // Upper/lower dot
                | '\u{05C7}'            // Qamats qatan
                | '\u{0591}'..='\u{05AF}' // Cantillation marks
            )
        })
        .collect()
}

/// Strips Unicode bidirectional control marks.
#[must_use]
pub fn strip_bidi_marks(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !matches!(c,
                '\u{200E}' | '\u{200F}' // LRM, RLM
                | '\u{061C}'            // ALM
                | '\u{202A}'..='\u{202E}' // Embedding/Override marks
                | '\u{2066}'..='\u{2069}' // Isolate marks
            )
        })
        .collect()
}

/// Normalizes Arabic text for matching.
///
/// - Alef variants (alef-madda, alef-hamza-above, alef-hamza-below) → bare Alef
/// - Teh marbuta → Heh
#[must_use]
pub fn normalize_arabic(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{0622}' | '\u{0623}' | '\u{0625}' => '\u{0627}', // Alef variants → Alef
            '\u{0629}' => '\u{0647}',                           // Teh marbuta → Heh
            _ => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arabic_diacritics() {
        // "kitāb" with fatha+kasra diacritics → strip them
        let with_harakat = "\u{0643}\u{0650}\u{062A}\u{064E}\u{0627}\u{0628}";
        let without = "\u{0643}\u{062A}\u{0627}\u{0628}";
        assert_eq!(strip_arabic_diacritics(with_harakat), without);
    }

    #[test]
    fn arabic_tatweel() {
        let with_tatweel = "\u{0639}\u{0640}\u{0631}\u{0628}\u{064A}";
        let without = "\u{0639}\u{0631}\u{0628}\u{064A}";
        assert_eq!(strip_arabic_diacritics(with_tatweel), without);
    }

    #[test]
    fn hebrew_diacritics() {
        // bet with dagesh → strip dagesh
        let with_niqqud = "\u{05D1}\u{05BC}";
        let without = "\u{05D1}";
        assert_eq!(strip_hebrew_diacritics(with_niqqud), without);
    }

    #[test]
    fn bidi_marks() {
        let with_bidi = "\u{200F}hello\u{200E} world";
        assert_eq!(strip_bidi_marks(with_bidi), "hello world");
    }

    #[test]
    fn arabic_normalization() {
        // Alef with hamza above → bare Alef
        assert_eq!(
            normalize_arabic("\u{0623}\u{062D}\u{0645}\u{062F}"),
            "\u{0627}\u{062D}\u{0645}\u{062F}"
        );
        // Teh marbuta → Heh
        assert_eq!(
            normalize_arabic("\u{0645}\u{062F}\u{0631}\u{0633}\u{0629}"),
            "\u{0645}\u{062F}\u{0631}\u{0633}\u{0647}"
        );
    }
}
