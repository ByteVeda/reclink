//! Cologne Phonetic (Kolner Phonetik) encoding algorithm.

use crate::phonetic::PhoneticEncoder;

/// Cologne Phonetic (Kolner Phonetik) is a phonetic algorithm optimized
/// for German names. It maps characters to digit codes based on context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ColognePhonetic;

impl PhoneticEncoder for ColognePhonetic {
    fn encode(&self, s: &str) -> String {
        cologne_phonetic(s)
    }
}

/// Maps a character to its Cologne digit code given context.
fn cologne_code(ch: char, prev: Option<char>, next: Option<char>) -> Option<char> {
    match ch {
        'A' | 'E' | 'I' | 'O' | 'U' | 'a' | 'e' | 'i' | 'o' | 'u' => Some('0'),
        'H' | 'h' => None, // H is ignored
        'B' | 'b' | 'P' | 'p' => {
            // P before H -> 3
            if ch == 'P' || ch == 'p' {
                if let Some(n) = next {
                    if n == 'H' || n == 'h' {
                        return Some('3');
                    }
                }
            }
            Some('1')
        }
        'D' | 'd' | 'T' | 't' => {
            // D/T before C/S/Z -> 8
            if let Some(n) = next {
                if "CSZcsz".contains(n) {
                    return Some('8');
                }
            }
            Some('2')
        }
        'F' | 'f' | 'V' | 'v' | 'W' | 'w' => Some('3'),
        'G' | 'g' | 'K' | 'k' | 'Q' | 'q' => Some('4'),
        'X' | 'x' => Some('4'), // X -> 48, but we handle this by returning '4' + appending '8'
        'L' | 'l' => Some('5'),
        'M' | 'm' | 'N' | 'n' => Some('6'),
        'R' | 'r' => Some('7'),
        'S' | 's' | 'Z' | 'z' => Some('8'),
        'C' | 'c' => {
            // C at start or after A/H/K/L/O/Q/R/U/X -> 4
            // C after S/Z -> 8
            // Otherwise -> 4 for first occurrence, 8 after vowels
            match prev {
                None => {
                    // Initial C: check next char
                    if let Some(n) = next {
                        if "AHKLOQRUXahkloqrux".contains(n) {
                            return Some('4');
                        }
                    }
                    Some('8')
                }
                Some(p) => {
                    if "SZsz".contains(p) {
                        Some('8')
                    } else if "AHKLOQRUXahkloqrux".contains(p) {
                        Some('4')
                    } else {
                        Some('8')
                    }
                }
            }
        }
        'J' | 'j' | 'Y' | 'y' => Some('0'),
        _ => None,
    }
}

/// Computes the Cologne Phonetic code for a string.
///
/// Returns a variable-length digit string. Returns an empty string for
/// empty or non-alphabetic input.
#[must_use]
pub fn cologne_phonetic(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }

    let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if chars.is_empty() {
        return String::new();
    }

    // Step 1: Map each character to its code
    let mut codes: Vec<char> = Vec::with_capacity(chars.len() * 2);
    for (i, &ch) in chars.iter().enumerate() {
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = chars.get(i + 1).copied();

        if let Some(code) = cologne_code(ch, prev, next) {
            codes.push(code);
        }

        // X produces '48'
        if ch == 'X' || ch == 'x' {
            codes.push('8');
        }
    }

    // Step 2: Remove consecutive duplicate codes
    let mut deduped: Vec<char> = Vec::with_capacity(codes.len());
    let mut last = '\0';
    for &code in &codes {
        if code != last {
            deduped.push(code);
            last = code;
        }
    }

    // Step 3: Remove all '0's except if it's the only character
    let result: String = if deduped.len() == 1 {
        deduped.iter().collect()
    } else {
        let first = deduped[0];
        let mut r = String::with_capacity(deduped.len());
        // Keep leading zero if present (it's the only case — initial vowel)
        if first != '0' {
            r.push(first);
        }
        for &code in &deduped[1..] {
            if code != '0' {
                r.push(code);
            }
        }
        if r.is_empty() {
            "0".to_string()
        } else {
            r
        }
    };

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert_eq!(cologne_phonetic(""), "");
    }

    #[test]
    fn known_values() {
        // Wikipedia examples for Cologne Phonetic
        assert_eq!(cologne_phonetic("Muller"), "657");
        assert_eq!(cologne_phonetic("Mueller"), "657");
    }

    #[test]
    fn german_names() {
        let c = ColognePhonetic;
        // Similar German names should match
        assert!(c.is_match("Meier", "Meyer"));
        assert!(c.is_match("Muller", "Mueller"));
    }

    #[test]
    fn non_matching() {
        let c = ColognePhonetic;
        assert!(!c.is_match("Schmidt", "Mueller"));
    }

    #[test]
    fn single_vowel() {
        let code = cologne_phonetic("A");
        assert_eq!(code, "0");
    }

    #[test]
    fn single_consonant() {
        let code = cologne_phonetic("B");
        assert_eq!(code, "1");
    }
}
