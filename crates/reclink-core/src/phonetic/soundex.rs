//! Soundex phonetic encoding algorithm.

use crate::phonetic::PhoneticEncoder;

/// Soundex encodes a name into a letter followed by three digits,
/// mapping similar-sounding consonants to the same digit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Soundex;

impl PhoneticEncoder for Soundex {
    fn encode(&self, s: &str) -> String {
        soundex(s)
    }
}

/// Maps a character to its Soundex digit, or `0` for vowels/H/W/Y.
fn soundex_code(c: char) -> Option<char> {
    match c.to_ascii_uppercase() {
        'B' | 'F' | 'P' | 'V' => Some('1'),
        'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some('2'),
        'D' | 'T' => Some('3'),
        'L' => Some('4'),
        'M' | 'N' => Some('5'),
        'R' => Some('6'),
        _ => None,
    }
}

/// Computes the Soundex code for a string.
///
/// Returns `"0000"` for empty input or strings with no letters.
#[must_use]
pub fn soundex(s: &str) -> String {
    let mut chars = s.chars().filter(|c| c.is_ascii_alphabetic());

    let first = match chars.next() {
        Some(c) => c.to_ascii_uppercase(),
        None => return "0000".to_string(),
    };

    let mut result = String::with_capacity(4);
    result.push(first);

    let mut last_code = soundex_code(first);

    for c in chars {
        if result.len() >= 4 {
            break;
        }
        let code = soundex_code(c);
        if let Some(digit) = code {
            if code != last_code {
                result.push(digit);
            }
        }
        // H and W don't separate identical codes, but vowels do
        if !c.eq_ignore_ascii_case(&'H') && !c.eq_ignore_ascii_case(&'W') {
            last_code = code;
        }
    }

    while result.len() < 4 {
        result.push('0');
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(soundex("Robert"), "R163");
        assert_eq!(soundex("Rupert"), "R163");
        assert_eq!(soundex("Ashcraft"), "A261");
        assert_eq!(soundex("Tymczak"), "T522");
    }

    #[test]
    fn empty_input() {
        assert_eq!(soundex(""), "0000");
    }

    #[test]
    fn single_letter() {
        assert_eq!(soundex("A"), "A000");
    }

    #[test]
    fn smith_variants() {
        assert_eq!(soundex("Smith"), "S530");
        assert_eq!(soundex("Smyth"), "S530");
    }

    #[test]
    fn is_match_test() {
        let s = Soundex;
        assert!(s.is_match("Smith", "Smyth"));
        assert!(!s.is_match("Smith", "Jones"));
    }
}
