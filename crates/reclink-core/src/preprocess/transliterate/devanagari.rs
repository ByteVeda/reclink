pub(super) fn transliterate_devanagari(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        // Independent vowels (0x0904-0x0914)
        let vowel = match ch {
            '\u{0904}' => Some("a"),  // Short A (Vedic)
            '\u{0905}' => Some("a"),  // A
            '\u{0906}' => Some("aa"), // Aa
            '\u{0907}' => Some("i"),  // I
            '\u{0908}' => Some("ii"), // Ii
            '\u{0909}' => Some("u"),  // U
            '\u{090A}' => Some("uu"), // Uu
            '\u{090B}' => Some("ri"), // Vocalic R
            '\u{090C}' => Some("li"), // Vocalic L
            '\u{090D}' => Some("e"),  // Candra E
            '\u{090E}' => Some("e"),  // Short E
            '\u{090F}' => Some("e"),  // E
            '\u{0910}' => Some("ai"), // Ai
            '\u{0911}' => Some("o"),  // Candra O
            '\u{0912}' => Some("o"),  // Short O
            '\u{0913}' => Some("o"),  // O
            '\u{0914}' => Some("au"), // Au
            _ => None,
        };

        if let Some(v) = vowel {
            out.push_str(v);
            i += 1;
            continue;
        }

        // Consonants (0x0915-0x0939)
        let consonant = match ch {
            '\u{0915}' => Some("k"),
            '\u{0916}' => Some("kh"),
            '\u{0917}' => Some("g"),
            '\u{0918}' => Some("gh"),
            '\u{0919}' => Some("ng"),
            '\u{091A}' => Some("ch"),
            '\u{091B}' => Some("chh"),
            '\u{091C}' => Some("j"),
            '\u{091D}' => Some("jh"),
            '\u{091E}' => Some("ny"),
            '\u{091F}' => Some("t"),
            '\u{0920}' => Some("th"),
            '\u{0921}' => Some("d"),
            '\u{0922}' => Some("dh"),
            '\u{0923}' => Some("n"),
            '\u{0924}' => Some("t"),
            '\u{0925}' => Some("th"),
            '\u{0926}' => Some("d"),
            '\u{0927}' => Some("dh"),
            '\u{0928}' => Some("n"),
            '\u{0929}' => Some("n"), // NNNA
            '\u{092A}' => Some("p"),
            '\u{092B}' => Some("ph"),
            '\u{092C}' => Some("b"),
            '\u{092D}' => Some("bh"),
            '\u{092E}' => Some("m"),
            '\u{092F}' => Some("y"),
            '\u{0930}' => Some("r"),
            '\u{0931}' => Some("r"), // RRA
            '\u{0932}' => Some("l"),
            '\u{0933}' => Some("l"), // LLA
            '\u{0934}' => Some("l"), // LLLA
            '\u{0935}' => Some("v"),
            '\u{0936}' => Some("sh"),
            '\u{0937}' => Some("sh"),
            '\u{0938}' => Some("s"),
            '\u{0939}' => Some("h"),
            _ => None,
        };

        if let Some(c) = consonant {
            out.push_str(c);
            i += 1;

            // Check what follows the consonant
            if i < chars.len() {
                let next = chars[i];
                // Virama (halant) — suppress inherent 'a'
                if next == '\u{094D}' {
                    i += 1;
                    continue;
                }
                // Dependent vowel signs (matras)
                let matra = match next {
                    '\u{093E}' => Some("a"),   // Aa matra
                    '\u{093F}' => Some("i"),   // I matra
                    '\u{0940}' => Some("ii"),  // Ii matra
                    '\u{0941}' => Some("u"),   // U matra
                    '\u{0942}' => Some("uu"),  // Uu matra
                    '\u{0943}' => Some("ri"),  // Vocalic R matra
                    '\u{0944}' => Some("rii"), // Vocalic RR matra
                    '\u{0945}' => Some("e"),   // Candra E matra
                    '\u{0946}' => Some("e"),   // Short E matra
                    '\u{0947}' => Some("e"),   // E matra
                    '\u{0948}' => Some("ai"),  // Ai matra
                    '\u{0949}' => Some("o"),   // Candra O matra
                    '\u{094A}' => Some("o"),   // Short O matra
                    '\u{094B}' => Some("o"),   // O matra
                    '\u{094C}' => Some("au"),  // Au matra
                    _ => None,
                };

                if let Some(m) = matra {
                    out.push_str(m);
                    i += 1;
                } else {
                    // No matra, no virama → inherent 'a'
                    out.push('a');
                }
            } else {
                // End of string → inherent 'a'
                out.push('a');
            }
            continue;
        }

        // Anusvara, visarga, chandrabindu
        match ch {
            '\u{0902}' => out.push('n'),      // Anusvara
            '\u{0903}' => out.push('h'),      // Visarga
            '\u{0901}' => out.push('n'),      // Chandrabindu
            '\u{0950}' => out.push_str("om"), // Om
            // Skip nukta and other combining marks
            '\u{093C}' | '\u{094D}' => {}
            _ => out.push(ch),
        }
        i += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use crate::preprocess::transliterate::{transliterate, Script};

    #[test]
    fn devanagari_basic() {
        // नमस्ते → namaste
        // न(na) म(ma) स(sa) ् (virama) ते(te)
        assert_eq!(
            transliterate(
                "\u{0928}\u{092E}\u{0938}\u{094D}\u{0924}\u{0947}",
                Script::Devanagari
            ),
            "namaste"
        );
    }

    #[test]
    fn devanagari_consonant_with_inherent_a() {
        // क → ka (consonant + inherent 'a')
        assert_eq!(transliterate("\u{0915}", Script::Devanagari), "ka");
    }

    #[test]
    fn devanagari_consonant_with_virama() {
        // क् → k (virama suppresses inherent 'a')
        assert_eq!(transliterate("\u{0915}\u{094D}", Script::Devanagari), "k");
    }

    #[test]
    fn devanagari_consonant_with_matra() {
        // कि → ki (consonant + i matra)
        assert_eq!(transliterate("\u{0915}\u{093F}", Script::Devanagari), "ki");
    }

    #[test]
    fn devanagari_independent_vowels() {
        // अ → a, इ → i
        assert_eq!(transliterate("\u{0905}", Script::Devanagari), "a");
        assert_eq!(transliterate("\u{0907}", Script::Devanagari), "i");
    }

    #[test]
    fn devanagari_mixed_script() {
        assert_eq!(
            transliterate("\u{0915}\u{092E} hello", Script::Devanagari),
            "kama hello"
        );
    }
}
