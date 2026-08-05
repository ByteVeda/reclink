pub(super) fn transliterate_cyrillic(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
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
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::preprocess::transliterate::{transliterate, Script};

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
}
