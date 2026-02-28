pub(super) fn transliterate_hebrew(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\u{05D0}' => out.push('a'),      // Alef
            '\u{05D1}' => out.push('b'),      // Bet
            '\u{05D2}' => out.push('g'),      // Gimel
            '\u{05D3}' => out.push('d'),      // Dalet
            '\u{05D4}' => out.push('h'),      // He
            '\u{05D5}' => out.push('v'),      // Vav
            '\u{05D6}' => out.push('z'),      // Zayin
            '\u{05D7}' => out.push_str("ch"), // Chet
            '\u{05D8}' => out.push('t'),      // Tet
            '\u{05D9}' => out.push('y'),      // Yod
            '\u{05DA}' => out.push('k'),      // Final Kaf
            '\u{05DB}' => out.push('k'),      // Kaf
            '\u{05DC}' => out.push('l'),      // Lamed
            '\u{05DD}' => out.push('m'),      // Final Mem
            '\u{05DE}' => out.push('m'),      // Mem
            '\u{05DF}' => out.push('n'),      // Final Nun
            '\u{05E0}' => out.push('n'),      // Nun
            '\u{05E1}' => out.push('s'),      // Samekh
            '\u{05E2}' => out.push('\''),     // Ayin
            '\u{05E3}' => out.push('p'),      // Final Pe
            '\u{05E4}' => out.push('p'),      // Pe
            '\u{05E5}' => out.push_str("ts"), // Final Tsadi
            '\u{05E6}' => out.push_str("ts"), // Tsadi
            '\u{05E7}' => out.push('q'),      // Qof
            '\u{05E8}' => out.push('r'),      // Resh
            '\u{05E9}' => out.push_str("sh"), // Shin
            '\u{05EA}' => out.push('t'),      // Tav
            // Skip niqqud and cantillation
            '\u{0591}'..='\u{05C7}' => {}
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::preprocess::transliterate::{transliterate, Script};

    #[test]
    fn hebrew_basic() {
        // שלום → shlvm
        assert_eq!(
            transliterate("\u{05E9}\u{05DC}\u{05D5}\u{05DD}", Script::Hebrew),
            "shlvm"
        );
    }

    #[test]
    fn hebrew_final_forms() {
        // Final kaf, mem, nun, pe, tsadi
        assert_eq!(transliterate("\u{05DA}", Script::Hebrew), "k"); // Final Kaf
        assert_eq!(transliterate("\u{05DD}", Script::Hebrew), "m"); // Final Mem
        assert_eq!(transliterate("\u{05DF}", Script::Hebrew), "n"); // Final Nun
    }

    #[test]
    fn hebrew_mixed_script() {
        assert_eq!(
            transliterate("\u{05D0}\u{05D1} hello", Script::Hebrew),
            "ab hello"
        );
    }
}
