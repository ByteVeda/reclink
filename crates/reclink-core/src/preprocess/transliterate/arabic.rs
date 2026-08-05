pub(super) fn transliterate_arabic(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\u{0621}' => out.push('\''),     // Hamza
            '\u{0622}' => out.push('a'),      // Alef Madda
            '\u{0623}' => out.push('a'),      // Alef Hamza Above
            '\u{0624}' => out.push('w'),      // Waw Hamza Above
            '\u{0625}' => out.push('a'),      // Alef Hamza Below
            '\u{0626}' => out.push('y'),      // Yeh Hamza Above
            '\u{0627}' => out.push('a'),      // Alef
            '\u{0628}' => out.push('b'),      // Ba
            '\u{0629}' => out.push('h'),      // Teh Marbuta
            '\u{062A}' => out.push('t'),      // Ta
            '\u{062B}' => out.push_str("th"), // Tha
            '\u{062C}' => out.push('j'),      // Jeem
            '\u{062D}' => out.push('h'),      // Ha
            '\u{062E}' => out.push_str("kh"), // Kha
            '\u{062F}' => out.push('d'),      // Dal
            '\u{0630}' => out.push_str("dh"), // Dhal
            '\u{0631}' => out.push('r'),      // Ra
            '\u{0632}' => out.push('z'),      // Zain
            '\u{0633}' => out.push('s'),      // Seen
            '\u{0634}' => out.push_str("sh"), // Sheen
            '\u{0635}' => out.push('s'),      // Sad
            '\u{0636}' => out.push('d'),      // Dad
            '\u{0637}' => out.push('t'),      // Tah
            '\u{0638}' => out.push('z'),      // Zah
            '\u{0639}' => out.push('\''),     // Ain
            '\u{063A}' => out.push_str("gh"), // Ghain
            '\u{0641}' => out.push('f'),      // Fa
            '\u{0642}' => out.push('q'),      // Qaf
            '\u{0643}' => out.push('k'),      // Kaf
            '\u{0644}' => out.push('l'),      // Lam
            '\u{0645}' => out.push('m'),      // Meem
            '\u{0646}' => out.push('n'),      // Noon
            '\u{0647}' => out.push('h'),      // Ha
            '\u{0648}' => out.push('w'),      // Waw
            '\u{0649}' => out.push('a'),      // Alef Maksura
            '\u{064A}' => out.push('y'),      // Ya
            // Skip diacritics
            '\u{064B}'..='\u{065F}' | '\u{0670}' | '\u{0640}' => {}
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::preprocess::transliterate::{transliterate, Script};

    #[test]
    fn arabic_basic() {
        // محمد → mhmd
        assert_eq!(
            transliterate("\u{0645}\u{062D}\u{0645}\u{062F}", Script::Arabic),
            "mhmd"
        );
    }

    #[test]
    fn arabic_mixed_script() {
        assert_eq!(
            transliterate("\u{0639}\u{0631}\u{0628} hello", Script::Arabic),
            "'rb hello"
        );
    }

    #[test]
    fn arabic_skips_diacritics() {
        // ba + fatha → b (diacritics stripped)
        let with_diacritics = "\u{0628}\u{064E}";
        assert_eq!(transliterate(with_diacritics, Script::Arabic), "b");
    }
}
