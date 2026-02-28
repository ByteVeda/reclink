pub(super) fn transliterate_hangul(s: &str) -> String {
    const HANGUL_BASE: u32 = 0xAC00;
    const HANGUL_END: u32 = 0xD7A3;

    const INITIAL: [&str; 19] = [
        "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t",
        "p", "h",
    ];
    const VOWEL: [&str; 21] = [
        "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo",
        "we", "wi", "yu", "eu", "ui", "i",
    ];
    const FINAL_CONSONANT: [&str; 28] = [
        "", "k", "kk", "ks", "n", "nj", "nh", "d", "l", "lk", "lm", "lb", "ls", "lt", "lp", "lh",
        "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h",
    ];

    // Jamo consonant table (for compatibility/leading jamo 0x3131-0x314E)
    const JAMO_CONSONANT: [(&str, char); 19] = [
        ("g", '\u{3131}'),
        ("kk", '\u{3132}'),
        ("n", '\u{3134}'),
        ("d", '\u{3137}'),
        ("tt", '\u{3138}'),
        ("r", '\u{3139}'),
        ("m", '\u{3141}'),
        ("b", '\u{3142}'),
        ("pp", '\u{3143}'),
        ("s", '\u{3145}'),
        ("ss", '\u{3146}'),
        ("", '\u{3147}'),
        ("j", '\u{3148}'),
        ("jj", '\u{3149}'),
        ("ch", '\u{314A}'),
        ("k", '\u{314B}'),
        ("t", '\u{314C}'),
        ("p", '\u{314D}'),
        ("h", '\u{314E}'),
    ];

    // Jamo vowel table (0x314F-0x3163)
    const JAMO_VOWEL: [(&str, char); 21] = [
        ("a", '\u{314F}'),
        ("ae", '\u{3150}'),
        ("ya", '\u{3151}'),
        ("yae", '\u{3152}'),
        ("eo", '\u{3153}'),
        ("e", '\u{3154}'),
        ("yeo", '\u{3155}'),
        ("ye", '\u{3156}'),
        ("o", '\u{3157}'),
        ("wa", '\u{3158}'),
        ("wae", '\u{3159}'),
        ("oe", '\u{315A}'),
        ("yo", '\u{315B}'),
        ("u", '\u{315C}'),
        ("wo", '\u{315D}'),
        ("we", '\u{315E}'),
        ("wi", '\u{315F}'),
        ("yu", '\u{3160}'),
        ("eu", '\u{3161}'),
        ("ui", '\u{3162}'),
        ("i", '\u{3163}'),
    ];

    let mut out = String::with_capacity(s.len() * 2);

    for ch in s.chars() {
        let cp = ch as u32;

        // Hangul syllable block: decompose algorithmically
        if (HANGUL_BASE..=HANGUL_END).contains(&cp) {
            let idx = cp - HANGUL_BASE;
            let initial = (idx / (21 * 28)) as usize;
            let medial = ((idx % (21 * 28)) / 28) as usize;
            let final_c = (idx % 28) as usize;

            out.push_str(INITIAL[initial]);
            out.push_str(VOWEL[medial]);
            out.push_str(FINAL_CONSONANT[final_c]);
            continue;
        }

        // Compatibility Jamo consonants
        if let Some((rom, _)) = JAMO_CONSONANT.iter().find(|(_, c)| *c == ch) {
            out.push_str(rom);
            continue;
        }

        // Compatibility Jamo vowels
        if let Some((rom, _)) = JAMO_VOWEL.iter().find(|(_, c)| *c == ch) {
            out.push_str(rom);
            continue;
        }

        // Pass through everything else
        out.push(ch);
    }

    out
}

#[cfg(test)]
mod tests {
    use crate::preprocess::transliterate::{transliterate, Script};

    #[test]
    fn hangul_basic() {
        // 한글 → hangul (한=h+a+n, 글=g+eu+l)
        assert_eq!(transliterate("한글", Script::Hangul), "hangeul");
    }

    #[test]
    fn hangul_seoul() {
        // 서울 → seooul → seo+ul (서=s+eo, 울=+u+l)
        assert_eq!(transliterate("서울", Script::Hangul), "seoul");
    }

    #[test]
    fn hangul_no_final() {
        // 가 → ga (가=g+a, no final consonant)
        assert_eq!(transliterate("가", Script::Hangul), "ga");
    }

    #[test]
    fn hangul_mixed_script() {
        assert_eq!(transliterate("한 hello", Script::Hangul), "han hello");
    }

    #[test]
    fn hangul_empty_initial() {
        // 아 → a (silent initial ieung + a)
        assert_eq!(transliterate("아", Script::Hangul), "a");
    }
}
