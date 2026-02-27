//! Transliteration from non-Latin scripts to Latin characters.
//!
//! Provides Cyrillic→Latin and Greek→Latin transliteration via static lookup tables.
//! Designed for preprocessing names and addresses in multilingual datasets.

/// Target script for transliteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    /// Cyrillic → Latin (ISO 9 / BGN/PCGN inspired).
    Cyrillic,
    /// Greek → Latin (UN/ELOT 743 inspired).
    Greek,
}

/// Transliterate a string from the given script to Latin characters.
///
/// Characters not in the lookup table are passed through unchanged,
/// so mixed-script text works correctly.
#[must_use]
pub fn transliterate(s: &str, script: Script) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match script {
            Script::Cyrillic => match ch {
                'А' => out.push('A'),
                'Б' => out.push('B'),
                'В' => out.push('V'),
                'Г' => out.push('G'),
                'Д' => out.push('D'),
                'Е' => out.push('E'),
                'Ё' => out.push_str("Yo"),
                'Ж' => out.push_str("Zh"),
                'З' => out.push('Z'),
                'И' => out.push('I'),
                'Й' => out.push('Y'),
                'К' => out.push('K'),
                'Л' => out.push('L'),
                'М' => out.push('M'),
                'Н' => out.push('N'),
                'О' => out.push('O'),
                'П' => out.push('P'),
                'Р' => out.push('R'),
                'С' => out.push('S'),
                'Т' => out.push('T'),
                'У' => out.push('U'),
                'Ф' => out.push('F'),
                'Х' => out.push_str("Kh"),
                'Ц' => out.push_str("Ts"),
                'Ч' => out.push_str("Ch"),
                'Ш' => out.push_str("Sh"),
                'Щ' => out.push_str("Shch"),
                'Ъ' => {} // hard sign — omit
                'Ы' => out.push('Y'),
                'Ь' => {} // soft sign — omit
                'Э' => out.push('E'),
                'Ю' => out.push_str("Yu"),
                'Я' => out.push_str("Ya"),
                'а' => out.push('a'),
                'б' => out.push('b'),
                'в' => out.push('v'),
                'г' => out.push('g'),
                'д' => out.push('d'),
                'е' => out.push('e'),
                'ё' => out.push_str("yo"),
                'ж' => out.push_str("zh"),
                'з' => out.push('z'),
                'и' => out.push('i'),
                'й' => out.push('y'),
                'к' => out.push('k'),
                'л' => out.push('l'),
                'м' => out.push('m'),
                'н' => out.push('n'),
                'о' => out.push('o'),
                'п' => out.push('p'),
                'р' => out.push('r'),
                'с' => out.push('s'),
                'т' => out.push('t'),
                'у' => out.push('u'),
                'ф' => out.push('f'),
                'х' => out.push_str("kh"),
                'ц' => out.push_str("ts"),
                'ч' => out.push_str("ch"),
                'ш' => out.push_str("sh"),
                'щ' => out.push_str("shch"),
                'ъ' => {} // hard sign — omit
                'ы' => out.push('y'),
                'ь' => {} // soft sign — omit
                'э' => out.push('e'),
                'ю' => out.push_str("yu"),
                'я' => out.push_str("ya"),
                // Ukrainian extras
                'Ґ' => out.push('G'),
                'ґ' => out.push('g'),
                'Є' => out.push_str("Ye"),
                'є' => out.push_str("ye"),
                'І' => out.push('I'),
                'і' => out.push('i'),
                'Ї' => out.push_str("Yi"),
                'ї' => out.push_str("yi"),
                _ => out.push(ch),
            },
            Script::Greek => match ch {
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
            },
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyrillic_basic() {
        assert_eq!(transliterate("Москва", Script::Cyrillic), "Moskva");
    }

    #[test]
    fn cyrillic_name() {
        assert_eq!(transliterate("Иванов", Script::Cyrillic), "Ivanov");
    }

    #[test]
    fn cyrillic_mixed_script() {
        assert_eq!(
            transliterate("Привет world", Script::Cyrillic),
            "Privet world"
        );
    }

    #[test]
    fn cyrillic_special_chars() {
        assert_eq!(transliterate("Щёлково", Script::Cyrillic), "Shchyolkovo");
    }

    #[test]
    fn cyrillic_hard_soft_signs() {
        assert_eq!(transliterate("объём", Script::Cyrillic), "obyom");
    }

    #[test]
    fn cyrillic_ukrainian() {
        // Київ = К+и+ї+в → K+i+yi+v
        assert_eq!(transliterate("Київ", Script::Cyrillic), "Kiyiv");
    }

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
