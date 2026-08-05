//! Match Rating Approach (MRA) phonetic algorithm.
//!
//! A two-part algorithm: encoding strips vowels (except leading) and
//! compresses duplicates, then comparison uses a minimum-rating threshold
//! based on combined code length.

use crate::phonetic::PhoneticEncoder;

/// Match Rating Approach phonetic encoder and comparator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MatchRatingApproach;

impl PhoneticEncoder for MatchRatingApproach {
    fn encode(&self, s: &str) -> String {
        mra_encode(s)
    }

    fn is_match(&self, a: &str, b: &str) -> bool {
        mra_compare(a, b)
    }
}

/// Encodes a string using the Match Rating Approach.
///
/// Steps:
/// 1. Remove non-alphabetic characters, uppercase
/// 2. Remove duplicate adjacent characters
/// 3. Remove vowels (except the first character)
/// 4. Return first 3 + last 3 characters (or full string if <= 6)
#[must_use]
pub fn mra_encode(s: &str) -> String {
    let upper: String = s
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return String::new();
    }

    // Remove duplicate adjacent characters
    let mut deduped = String::with_capacity(upper.len());
    let mut last = '\0';
    for c in upper.chars() {
        if c != last {
            deduped.push(c);
        }
        last = c;
    }

    // Remove vowels except the first character
    let chars: Vec<char> = deduped.chars().collect();
    let mut stripped = String::with_capacity(chars.len());
    if let Some(&first) = chars.first() {
        stripped.push(first);
    }
    for &c in &chars[1..] {
        if !is_vowel(c) {
            stripped.push(c);
        }
    }

    // Take first 3 + last 3 (or full if <= 6)
    let code_chars: Vec<char> = stripped.chars().collect();
    if code_chars.len() <= 6 {
        return stripped;
    }

    let mut result = String::with_capacity(6);
    for &c in &code_chars[..3] {
        result.push(c);
    }
    for &c in &code_chars[code_chars.len() - 3..] {
        result.push(c);
    }
    result
}

/// Compares two strings using the Match Rating Approach.
///
/// Returns `true` if the strings are considered a phonetic match.
#[must_use]
pub fn mra_compare(a: &str, b: &str) -> bool {
    let code_a = mra_encode(a);
    let code_b = mra_encode(b);

    if code_a.is_empty() || code_b.is_empty() {
        return code_a.is_empty() && code_b.is_empty();
    }

    let len_a = code_a.len();
    let len_b = code_b.len();

    // If length difference > 3, not a match
    if len_a.abs_diff(len_b) > 3 {
        return false;
    }

    // Minimum rating based on combined length
    let combined = len_a + len_b;
    let min_rating = match combined {
        0..=4 => 5,
        5..=7 => 4,
        8..=11 => 3,
        _ => 2,
    };

    // Compare from left, removing matches
    let a_chars: Vec<char> = code_a.chars().collect();
    let b_chars: Vec<char> = code_b.chars().collect();

    let mut a_remaining: Vec<Option<char>> = a_chars.iter().copied().map(Some).collect();
    let mut b_remaining: Vec<Option<char>> = b_chars.iter().copied().map(Some).collect();

    // Left-to-right pass
    for a_slot in &mut a_remaining {
        if let Some(ac) = *a_slot {
            for b_slot in &mut b_remaining {
                if let Some(bc) = *b_slot {
                    if ac == bc {
                        *a_slot = None;
                        *b_slot = None;
                        break;
                    }
                }
            }
        }
    }

    // Right-to-left pass
    let a_left: Vec<char> = a_remaining.iter().filter_map(|c| *c).collect();
    let b_left: Vec<char> = b_remaining.iter().filter_map(|c| *c).collect();

    let mut unmatched = 0;
    let mut a_rev = a_left.iter().rev();
    let mut b_rev = b_left.iter().rev();

    loop {
        match (a_rev.next(), b_rev.next()) {
            (Some(ac), Some(bc)) => {
                if ac != bc {
                    unmatched += 2;
                }
            }
            (Some(_), None) | (None, Some(_)) => {
                unmatched += 1;
            }
            (None, None) => break,
        }
    }

    6 - unmatched >= min_rating
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'A' | 'E' | 'I' | 'O' | 'U')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_known_values() {
        assert_eq!(mra_encode("Smith"), "SMTH");
        assert_eq!(mra_encode("Catherine"), "CTHRN");
    }

    #[test]
    fn encode_empty() {
        assert_eq!(mra_encode(""), "");
    }

    #[test]
    fn encode_single() {
        assert_eq!(mra_encode("A"), "A");
    }

    #[test]
    fn compare_similar_names() {
        assert!(mra_compare("Smith", "Smyth"));
        assert!(mra_compare("Catherine", "Kathryn"));
    }

    #[test]
    fn compare_different_names() {
        assert!(!mra_compare("Smith", "Jones"));
    }

    #[test]
    fn compare_empty() {
        assert!(mra_compare("", ""));
        assert!(!mra_compare("Smith", ""));
    }

    #[test]
    fn is_match_trait() {
        let mra = MatchRatingApproach;
        assert!(mra.is_match("Smith", "Smyth"));
    }

    #[test]
    fn encode_duplicate_removal() {
        // "Llyod" -> deduplicate L -> "Loyd" -> strip vowels -> "LYD"
        let code = mra_encode("Lloyd");
        assert!(!code.contains("LL"));
    }
}
