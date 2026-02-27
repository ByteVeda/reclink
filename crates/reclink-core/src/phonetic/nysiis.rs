//! NYSIIS (New York State Identification and Intelligence System) phonetic algorithm.

use crate::phonetic::PhoneticEncoder;

/// NYSIIS produces phonetic codes that are generally considered more accurate
/// than Soundex for matching names.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Nysiis;

impl PhoneticEncoder for Nysiis {
    fn encode(&self, s: &str) -> String {
        nysiis(s)
    }
}

/// Computes the NYSIIS encoding of a string.
#[must_use]
pub fn nysiis(s: &str) -> String {
    let upper: String = s
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return String::new();
    }

    let mut name: Vec<char> = upper.chars().collect();

    // Step 1: Translate first characters
    if name.starts_with(&['M', 'A', 'C']) {
        name.splice(0..3, "MCC".chars());
    } else if name.starts_with(&['K', 'N']) {
        name.remove(0);
    } else if name.starts_with(&['K']) {
        name[0] = 'C';
    } else if name.starts_with(&['P', 'H']) || name.starts_with(&['P', 'F']) {
        name[0] = 'F';
        name[1] = 'F';
    } else if name.starts_with(&['S', 'C', 'H']) {
        name[1] = 'S';
        name[2] = 'S';
    }

    // Step 2: Translate last characters
    let len = name.len();
    if len >= 2 {
        if name.ends_with(&['E', 'E']) || name.ends_with(&['I', 'E']) {
            name.truncate(len - 2);
            name.push('Y');
        } else if name.ends_with(&['D', 'T'])
            || name.ends_with(&['R', 'T'])
            || name.ends_with(&['R', 'D'])
            || name.ends_with(&['N', 'T'])
            || name.ends_with(&['N', 'D'])
        {
            name.truncate(len - 2);
            name.push('D');
        }
    }

    if name.is_empty() {
        return String::new();
    }

    let first_char = name[0];
    let mut result = String::new();
    result.push(first_char);

    let is_vowel = |c: char| matches!(c, 'A' | 'E' | 'I' | 'O' | 'U');

    let mut i = 1;
    while i < name.len() {
        let c = name[i];
        let next = if i + 1 < name.len() {
            name[i + 1]
        } else {
            '\0'
        };

        let replacement = if c == 'E' && next == 'V' {
            i += 1;
            Some("AF")
        } else if is_vowel(c) {
            Some("A")
        } else if c == 'Q' {
            Some("G")
        } else if c == 'Z' {
            Some("S")
        } else if c == 'M' {
            Some("N")
        } else if c == 'K' {
            if next == 'N' {
                Some("N")
            } else {
                Some("C")
            }
        } else if c == 'S' && next == 'C' && i + 2 < name.len() && name[i + 2] == 'H' {
            i += 2;
            Some("SSS")
        } else if c == 'P' && next == 'H' {
            i += 1;
            Some("FF")
        } else if c == 'H' && (!is_vowel(name[i - 1]) || (i + 1 < name.len() && !is_vowel(next))) {
            let prev = name[i - 1];
            result.push(if is_vowel(prev) { 'A' } else { prev });
            i += 1;
            continue;
        } else if c == 'W' && is_vowel(name[i - 1]) {
            let prev = name[i - 1];
            result.push(prev);
            i += 1;
            continue;
        } else {
            result.push(c);
            i += 1;
            continue;
        };

        if let Some(rep) = replacement {
            for ch in rep.chars() {
                result.push(ch);
            }
        }

        i += 1;
    }

    // Remove trailing S
    if result.ends_with('S') && result.len() > 1 {
        result.pop();
    }

    // Replace trailing AY with Y
    if result.ends_with("AY") {
        result.truncate(result.len() - 2);
        result.push('Y');
    }

    // Remove trailing A
    if result.ends_with('A') && result.len() > 1 {
        result.pop();
    }

    // Collapse consecutive duplicate characters
    let chars: Vec<char> = result.chars().collect();
    let mut deduped = String::new();
    for (j, &ch) in chars.iter().enumerate() {
        if j == 0 || ch != chars[j - 1] {
            deduped.push(ch);
        }
    }

    // Truncate to 6 characters
    deduped.truncate(6);
    deduped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(nysiis("Johnson"), "JANSAN");
        assert_eq!(nysiis("Smith"), "SNAT");
    }

    #[test]
    fn empty_input() {
        assert_eq!(nysiis(""), "");
    }

    #[test]
    fn mac_prefix() {
        let code = nysiis("MacDonald");
        assert!(code.starts_with('M'));
    }

    #[test]
    fn is_match_test() {
        let n = Nysiis;
        assert!(n.is_match("Johnson", "Jonson"));
    }
}
