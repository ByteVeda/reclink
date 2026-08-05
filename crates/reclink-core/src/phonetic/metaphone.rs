//! Metaphone phonetic encoding algorithm.

use crate::phonetic::PhoneticEncoder;

/// Metaphone produces a variable-length phonetic key that represents
/// the English pronunciation of a word.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Metaphone;

impl PhoneticEncoder for Metaphone {
    fn encode(&self, s: &str) -> String {
        metaphone(s)
    }
}

/// Computes the Metaphone encoding of a string.
#[must_use]
pub fn metaphone(s: &str) -> String {
    let upper: String = s
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return String::new();
    }

    let chars: Vec<char> = upper.chars().collect();
    let len = chars.len();
    let mut result = String::new();
    let mut i = 0;

    // Drop initial silent letters
    if len >= 2 {
        match (chars[0], chars[1]) {
            ('A', 'E') | ('G', 'N') | ('K', 'N') | ('P', 'N') | ('W', 'R') => i = 1,
            _ => {}
        }
    }

    let at = |idx: usize| -> char {
        if idx < len {
            chars[idx]
        } else {
            '\0'
        }
    };

    let is_vowel = |c: char| matches!(c, 'A' | 'E' | 'I' | 'O' | 'U');

    while i < len {
        let c = chars[i];

        // Skip duplicate adjacent letters (except C)
        if c != 'C' && i > 0 && chars[i - 1] == c {
            i += 1;
            continue;
        }

        match c {
            'A' | 'E' | 'I' | 'O' | 'U' => {
                if i == 0 {
                    result.push(c);
                }
            }
            'B' => {
                if !(i > 0 && chars[i - 1] == 'M') || i + 1 >= len {
                    result.push('B');
                }
            }
            'C' => {
                if at(i + 1) == 'I' && at(i + 2) == 'A' {
                    result.push('X');
                } else if matches!(at(i + 1), 'E' | 'I' | 'Y') {
                    result.push('S');
                } else if at(i + 1) == 'H' {
                    result.push('X');
                    i += 1;
                } else {
                    result.push('K');
                }
            }
            'D' => {
                if at(i + 1) == 'G' && matches!(at(i + 2), 'E' | 'I' | 'Y') {
                    result.push('J');
                } else {
                    result.push('T');
                }
            }
            'F' => result.push('F'),
            'G' => {
                if i + 1 < len && at(i + 1) == 'H' && i + 2 < len && !is_vowel(at(i + 2)) {
                    i += 1;
                } else if i > 0 && at(i + 1) == 'N' {
                    // silent
                } else if i > 0 && chars[i - 1] == 'G' {
                    // skip
                } else {
                    let next = at(i + 1);
                    if matches!(next, 'E' | 'I' | 'Y') {
                        result.push('J');
                    } else {
                        result.push('K');
                    }
                }
            }
            'H' => {
                if is_vowel(at(i + 1)) && (i == 0 || !is_vowel(chars[i - 1])) {
                    result.push('H');
                }
            }
            'J' => result.push('J'),
            'K' => {
                if i == 0 || chars[i - 1] != 'C' {
                    result.push('K');
                }
            }
            'L' => result.push('L'),
            'M' => result.push('M'),
            'N' => result.push('N'),
            'P' => {
                if at(i + 1) == 'H' {
                    result.push('F');
                    i += 1;
                } else {
                    result.push('P');
                }
            }
            'Q' => result.push('K'),
            'R' => result.push('R'),
            'S' => {
                if at(i + 1) == 'H' || (at(i + 1) == 'I' && matches!(at(i + 2), 'A' | 'O')) {
                    result.push('X');
                    i += 1;
                } else {
                    result.push('S');
                }
            }
            'T' => {
                if at(i + 1) == 'H' {
                    result.push('0');
                    i += 1;
                } else if at(i + 1) == 'I' && matches!(at(i + 2), 'A' | 'O') {
                    result.push('X');
                } else {
                    result.push('T');
                }
            }
            'V' => result.push('F'),
            'W' | 'Y' => {
                if is_vowel(at(i + 1)) {
                    result.push(c);
                }
            }
            'X' => {
                result.push('K');
                result.push('S');
            }
            'Z' => result.push('S'),
            _ => {}
        }

        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(metaphone("Smith"), "SM0");
        assert_eq!(metaphone("Schmidt"), "SXMTT");
    }

    #[test]
    fn empty_input() {
        assert_eq!(metaphone(""), "");
    }

    #[test]
    fn silent_initial() {
        assert_eq!(metaphone("Knight"), "NT");
        assert_eq!(metaphone("Gnome"), "NM");
    }
}
