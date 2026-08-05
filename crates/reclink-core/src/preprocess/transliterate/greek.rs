pub(super) fn transliterate_greek(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'Α' => out.push('A'),
            'Β' => out.push('V'),
            'Γ' => out.push('G'),
            'Δ' => out.push('D'),
            'Ε' => out.push('E'),
            'Ζ' => out.push('Z'),
            'Η' => out.push('I'),
            'Θ' => out.push_str("Th"),
            'Ι' => out.push('I'),
            'Κ' => out.push('K'),
            'Λ' => out.push('L'),
            'Μ' => out.push('M'),
            'Ν' => out.push('N'),
            'Ξ' => out.push_str("Ks"),
            'Ο' => out.push('O'),
            'Π' => out.push('P'),
            'Ρ' => out.push('R'),
            'Σ' => out.push('S'),
            'Τ' => out.push('T'),
            'Υ' => out.push('Y'),
            'Φ' => out.push_str("Ph"),
            'Χ' => out.push_str("Ch"),
            'Ψ' => out.push_str("Ps"),
            'Ω' => out.push('O'),
            'α' => out.push('a'),
            'β' => out.push('v'),
            'γ' => out.push('g'),
            'δ' => out.push('d'),
            'ε' => out.push('e'),
            'ζ' => out.push('z'),
            'η' => out.push('i'),
            'θ' => out.push_str("th"),
            'ι' => out.push('i'),
            'κ' => out.push('k'),
            'λ' => out.push('l'),
            'μ' => out.push('m'),
            'ν' => out.push('n'),
            'ξ' => out.push_str("ks"),
            'ο' => out.push('o'),
            'π' => out.push('p'),
            'ρ' => out.push('r'),
            'σ' => out.push('s'),
            'ς' => out.push('s'), // final sigma
            'τ' => out.push('t'),
            'υ' => out.push('y'),
            'φ' => out.push_str("ph"),
            'χ' => out.push_str("ch"),
            'ψ' => out.push_str("ps"),
            'ω' => out.push('o'),
            // Accented vowels (strip accent, map to base)
            'Ά' => out.push('A'),
            'Έ' => out.push('E'),
            'Ή' => out.push('I'),
            'Ί' => out.push('I'),
            'Ό' => out.push('O'),
            'Ύ' => out.push('Y'),
            'Ώ' => out.push('O'),
            'ά' => out.push('a'),
            'έ' => out.push('e'),
            'ή' => out.push('i'),
            'ί' => out.push('i'),
            'ό' => out.push('o'),
            'ύ' => out.push('y'),
            'ώ' => out.push('o'),
            'ϊ' => out.push('i'),
            'ϋ' => out.push('y'),
            'ΐ' => out.push('i'),
            'ΰ' => out.push('y'),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::preprocess::transliterate::{transliterate, Script};

    #[test]
    fn greek_basic() {
        assert_eq!(transliterate("Αθήνα", Script::Greek), "Athina");
    }

    #[test]
    fn greek_name() {
        assert_eq!(transliterate("Παπαδόπουλος", Script::Greek), "Papadopoylos");
    }

    #[test]
    fn greek_final_sigma() {
        assert_eq!(transliterate("λόγος", Script::Greek), "logos");
    }

    #[test]
    fn greek_mixed_script() {
        assert_eq!(
            transliterate("Ελλάδα Greece", Script::Greek),
            "Ellada Greece"
        );
    }
}
