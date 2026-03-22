//! Phonex phonetic encoding algorithm.
//!
//! An improved variant of Soundex with better handling of silent letters,
//! common prefixes, and vowel groups.

use crate::phonetic::PhoneticEncoder;

/// Phonex produces an improved Soundex-like code with better prefix handling
/// and phoneme groupings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Phonex;

impl PhoneticEncoder for Phonex {
    fn encode(&self, s: &str) -> String {
        phonex(s)
    }
}

/// Maps a character to its Phonex digit.
fn phonex_code(c: char) -> Option<char> {
    match c {
        'B' | 'F' | 'P' | 'V' => Some('1'),
        'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some('2'),
        'D' | 'T' => Some('3'),
        'L' => Some('4'),
        'M' | 'N' => Some('5'),
        'R' => Some('6'),
        _ => None,
    }
}

/// Computes the Phonex code for a string.
///
/// Returns `"0000"` for empty input or strings with no letters.
#[must_use]
pub fn phonex(s: &str) -> String {
    let upper: String = s
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return "0000".to_string();
    }

    let chars: Vec<char> = upper.chars().collect();

    // Pre-process: handle common prefixes
    let start = preprocess_prefix(&chars);
    if start >= chars.len() {
        let mut result = String::with_capacity(4);
        result.push(chars[0]);
        while result.len() < 4 {
            result.push('0');
        }
        return result;
    }

    let first = chars[start];

    // Handle initial vowel: treat as 'V' for coding purposes
    let first_char = if "AEIOUY".contains(first) { 'V' } else { first };

    let mut result = String::with_capacity(4);
    result.push(first_char);

    let mut last_code = phonex_code(first_char);

    for &c in &chars[start + 1..] {
        if result.len() >= 4 {
            break;
        }
        let code = phonex_code(c);
        if let Some(digit) = code {
            if code != last_code {
                result.push(digit);
            }
        }
        // H and W don't separate identical codes, vowels do
        if c != 'H' && c != 'W' {
            last_code = code;
        }
    }

    while result.len() < 4 {
        result.push('0');
    }

    result
}

/// Handle common prefixes, returning the index to start encoding from.
fn preprocess_prefix(chars: &[char]) -> usize {
    let len = chars.len();
    if len >= 2 {
        let pair = (chars[0], chars[1]);
        match pair {
            ('P', 'H') => return 1, // PH -> treat as F, skip P
            ('P', 'F') => return 1, // PF -> treat as F, skip P
            ('K', 'N') => return 1, // KN -> skip K
            ('W', 'R') => return 1, // WR -> skip W
            ('A', 'E') => return 1, // AE -> skip A
            _ => {}
        }
    }
    if len >= 3 && chars[0] == 'S' && chars[1] == 'C' && chars[2] == 'H' {
        return 1; // SCH -> skip S
    }
    // Drop leading H, W, Y
    if len >= 1 && matches!(chars[0], 'H' | 'W' | 'Y') {
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // Phonex should produce reasonable codes for common names
        assert_eq!(phonex("Smith"), "S530");
        assert_eq!(phonex("Smyth"), "S530");
    }

    #[test]
    fn prefix_handling() {
        // KN prefix: K is dropped
        let code = phonex("Knight");
        assert!(code.starts_with('N'));

        // WR prefix: W is dropped
        let code = phonex("Wright");
        assert!(code.starts_with('R'));

        // PH prefix: P is dropped, H treated as start
        let code = phonex("Pham");
        assert!(code.starts_with('H'));
    }

    #[test]
    fn empty_input() {
        assert_eq!(phonex(""), "0000");
    }

    #[test]
    fn single_letter() {
        assert_eq!(phonex("A"), "V000");
    }

    #[test]
    fn vowel_start() {
        // Initial vowels are mapped to 'V'
        let code = phonex("Anderson");
        assert!(code.starts_with('V'));
    }

    #[test]
    fn is_match_test() {
        let p = Phonex;
        assert!(p.is_match("Smith", "Smyth"));
    }
}
