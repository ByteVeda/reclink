//! Caverphone 2 phonetic encoding algorithm.

use crate::phonetic::PhoneticEncoder;

/// Caverphone 2 is a phonetic algorithm developed for matching New Zealand
/// names. It produces a 10-character code.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Caverphone;

impl PhoneticEncoder for Caverphone {
    fn encode(&self, s: &str) -> String {
        caverphone(s)
    }
}

/// Computes the Caverphone 2 code for a string.
///
/// Returns a 10-character uppercase code, padded with '1's.
#[must_use]
pub fn caverphone(s: &str) -> String {
    if s.is_empty() {
        return "1111111111".to_string();
    }

    // Step 1: lowercase and remove non-alpha
    let mut result: String = s
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_lowercase())
        .collect();

    if result.is_empty() {
        return "1111111111".to_string();
    }

    // Step 2: Remove trailing 'e'
    if result.ends_with('e') {
        result.pop();
    }
    if result.is_empty() {
        return "1111111111".to_string();
    }

    // Step 3: Apply ordered replacement rules
    let replacements: &[(&str, &str)] = &[
        ("cough", "cof2"),
        ("rough", "rof2"),
        ("tough", "tof2"),
        ("enough", "enof2"),
        ("trough", "trof2"),
        ("gn", "2n"),
        ("mb", "m2"),
        ("cq", "2q"),
        ("ci", "si"),
        ("ce", "se"),
        ("cy", "sy"),
        ("tch", "2ch"),
        ("c", "k"),
        ("q", "k"),
        ("x", "k"),
        ("v", "f"),
        ("dg", "2g"),
        ("tio", "sio"),
        ("tia", "sia"),
        ("d", "t"),
        ("ph", "fh"),
        ("b", "p"),
        ("sh", "s2"),
        ("z", "s"),
    ];

    for &(from, to) in replacements {
        result = result.replace(from, to);
    }

    // Step 4: Replace initial vowel with 'A', other vowels with '3'
    let chars: Vec<char> = result.chars().collect();
    let mut new_result = String::with_capacity(chars.len());
    for (i, &ch) in chars.iter().enumerate() {
        if "aeiou".contains(ch) {
            if i == 0 {
                new_result.push('A');
            } else {
                new_result.push('3');
            }
        } else {
            new_result.push(ch);
        }
    }
    result = new_result;

    // Step 5: Remove '3's, 'h' after a non-vowel, and '2's
    let chars: Vec<char> = result.chars().collect();
    let mut cleaned = String::with_capacity(chars.len());
    for (i, &ch) in chars.iter().enumerate() {
        if ch == '3' || ch == '2' {
            continue;
        }
        if ch == 'h' && i > 0 && !"AEIOUaeiou3".contains(chars[i - 1]) {
            continue;
        }
        cleaned.push(ch);
    }
    result = cleaned;

    // Step 6: Remove consecutive duplicate characters
    let mut deduped = String::with_capacity(result.len());
    let mut last = '\0';
    for ch in result.chars() {
        if ch != last {
            deduped.push(ch);
            last = ch;
        }
    }

    // Step 7: Uppercase, pad or truncate to 10 characters
    let mut result: String = deduped.to_uppercase();
    while result.len() < 10 {
        result.push('1');
    }
    result.truncate(10);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert_eq!(caverphone(""), "1111111111");
    }

    #[test]
    fn known_values() {
        assert_eq!(caverphone("Lee"), "L111111111");
        assert_eq!(caverphone("David"), "TFT1111111");
    }

    #[test]
    fn matching_names() {
        let c = Caverphone;
        // Similar-sounding names should match
        assert!(c.is_match("Lee", "Lea"));
    }

    #[test]
    fn non_matching() {
        let c = Caverphone;
        assert!(!c.is_match("Smith", "Jones"));
    }

    #[test]
    fn single_letter() {
        let code = caverphone("A");
        assert_eq!(code.len(), 10);
        assert_eq!(code, "A111111111");
    }

    #[test]
    fn ten_char_output() {
        assert_eq!(caverphone("test").len(), 10);
        assert_eq!(caverphone("a").len(), 10);
        assert_eq!(caverphone("Christopher").len(), 10);
    }
}
