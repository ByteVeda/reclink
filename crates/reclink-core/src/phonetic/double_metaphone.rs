//! Double Metaphone phonetic encoding algorithm.
//!
//! Returns a primary and alternate encoding to handle words with
//! multiple possible pronunciations.

use crate::phonetic::PhoneticEncoder;

/// Double Metaphone produces primary and alternate phonetic codes,
/// better handling non-English origins than standard Metaphone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DoubleMetaphone;

impl PhoneticEncoder for DoubleMetaphone {
    fn encode(&self, s: &str) -> String {
        double_metaphone(s).0
    }
}

impl DoubleMetaphone {
    /// Returns both primary and alternate encodings.
    #[must_use]
    pub fn encode_both(&self, s: &str) -> (String, String) {
        double_metaphone(s)
    }
}

/// Computes Double Metaphone encoding, returning (primary, alternate).
///
/// This is a simplified implementation covering the most common cases.
#[must_use]
pub fn double_metaphone(s: &str) -> (String, String) {
    let upper: Vec<char> = s
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return (String::new(), String::new());
    }

    let len = upper.len();
    let mut primary = String::new();
    let mut alternate = String::new();
    let mut i = 0;
    let max_len = 4;

    let at = |idx: usize| -> char {
        if idx < len {
            upper[idx]
        } else {
            '\0'
        }
    };

    let is_vowel = |c: char| matches!(c, 'A' | 'E' | 'I' | 'O' | 'U');

    // Skip initial silent letters
    if len >= 2
        && matches!(
            (upper[0], upper[1]),
            ('G', 'N') | ('K', 'N') | ('P', 'N') | ('A', 'E') | ('W', 'R')
        )
    {
        i = 1;
    }

    while i < len && (primary.len() < max_len || alternate.len() < max_len) {
        let c = at(i);

        match c {
            'A' | 'E' | 'I' | 'O' | 'U' => {
                if i == 0 {
                    primary.push('A');
                    alternate.push('A');
                }
                i += 1;
            }
            'B' => {
                primary.push('P');
                alternate.push('P');
                i += if at(i + 1) == 'B' { 2 } else { 1 };
            }
            'C' => {
                if at(i + 1) == 'H' {
                    primary.push('X');
                    alternate.push('X');
                    i += 2;
                } else if matches!(at(i + 1), 'I' | 'E' | 'Y') {
                    primary.push('S');
                    alternate.push('S');
                    i += 1;
                } else {
                    primary.push('K');
                    alternate.push('K');
                    i += if at(i + 1) == 'C' { 2 } else { 1 };
                }
            }
            'D' => {
                if at(i + 1) == 'G' && matches!(at(i + 2), 'I' | 'E' | 'Y') {
                    primary.push('J');
                    alternate.push('J');
                    i += 2;
                } else {
                    primary.push('T');
                    alternate.push('T');
                    i += if at(i + 1) == 'D' { 2 } else { 1 };
                }
            }
            'F' => {
                primary.push('F');
                alternate.push('F');
                i += if at(i + 1) == 'F' { 2 } else { 1 };
            }
            'G' => {
                if at(i + 1) == 'H' {
                    if i + 2 < len && is_vowel(at(i + 2)) {
                        primary.push('K');
                        alternate.push('K');
                    }
                    i += 2;
                } else if matches!(at(i + 1), 'E' | 'I' | 'Y') {
                    primary.push('J');
                    alternate.push('K');
                    i += 1;
                } else {
                    primary.push('K');
                    alternate.push('K');
                    i += if at(i + 1) == 'G' { 2 } else { 1 };
                }
            }
            'H' => {
                if is_vowel(at(i + 1)) && (i == 0 || !is_vowel(at(i.wrapping_sub(1)))) {
                    primary.push('H');
                    alternate.push('H');
                }
                i += 1;
            }
            'J' => {
                primary.push('J');
                alternate.push('H');
                i += if at(i + 1) == 'J' { 2 } else { 1 };
            }
            'K' => {
                primary.push('K');
                alternate.push('K');
                i += if at(i + 1) == 'K' { 2 } else { 1 };
            }
            'L' => {
                primary.push('L');
                alternate.push('L');
                i += if at(i + 1) == 'L' { 2 } else { 1 };
            }
            'M' => {
                primary.push('M');
                alternate.push('M');
                i += if at(i + 1) == 'M' { 2 } else { 1 };
            }
            'N' => {
                primary.push('N');
                alternate.push('N');
                i += if at(i + 1) == 'N' { 2 } else { 1 };
            }
            'P' => {
                if at(i + 1) == 'H' {
                    primary.push('F');
                    alternate.push('F');
                    i += 2;
                } else {
                    primary.push('P');
                    alternate.push('P');
                    i += if at(i + 1) == 'P' { 2 } else { 1 };
                }
            }
            'Q' => {
                primary.push('K');
                alternate.push('K');
                i += if at(i + 1) == 'Q' { 2 } else { 1 };
            }
            'R' => {
                primary.push('R');
                alternate.push('R');
                i += if at(i + 1) == 'R' { 2 } else { 1 };
            }
            'S' => {
                if at(i + 1) == 'H' {
                    primary.push('X');
                    alternate.push('X');
                    i += 2;
                } else if at(i + 1) == 'C' && matches!(at(i + 2), 'I' | 'E' | 'Y') {
                    primary.push('S');
                    alternate.push('S');
                    i += 3;
                } else {
                    primary.push('S');
                    alternate.push('S');
                    i += if at(i + 1) == 'S' { 2 } else { 1 };
                }
            }
            'T' => {
                if at(i + 1) == 'H' {
                    primary.push('T');
                    alternate.push('0');
                    i += 2;
                } else {
                    primary.push('T');
                    alternate.push('T');
                    i += if at(i + 1) == 'T' { 2 } else { 1 };
                }
            }
            'V' => {
                primary.push('F');
                alternate.push('F');
                i += if at(i + 1) == 'V' { 2 } else { 1 };
            }
            'W' => {
                if is_vowel(at(i + 1)) {
                    primary.push('A');
                    alternate.push('A');
                }
                i += 1;
            }
            'X' => {
                primary.push_str("KS");
                alternate.push_str("KS");
                i += if at(i + 1) == 'X' { 2 } else { 1 };
            }
            'Y' => {
                if is_vowel(at(i + 1)) {
                    primary.push('A');
                    alternate.push('A');
                }
                i += 1;
            }
            'Z' => {
                primary.push('S');
                alternate.push('S');
                i += if at(i + 1) == 'Z' { 2 } else { 1 };
            }
            _ => {
                i += 1;
            }
        }
    }

    primary.truncate(max_len);
    alternate.truncate(max_len);

    (primary, alternate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_encoding() {
        let (primary, _alt) = double_metaphone("Smith");
        assert!(!primary.is_empty());
    }

    #[test]
    fn empty_input() {
        assert_eq!(double_metaphone(""), (String::new(), String::new()));
    }

    #[test]
    fn encode_trait() {
        let dm = DoubleMetaphone;
        assert!(!dm.encode("John").is_empty());
    }

    #[test]
    fn both_codes() {
        let dm = DoubleMetaphone;
        let (primary, alternate) = dm.encode_both("Thomas");
        assert!(!primary.is_empty());
        assert!(!alternate.is_empty());
    }
}
