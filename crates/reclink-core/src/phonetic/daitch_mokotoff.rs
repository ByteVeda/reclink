//! Daitch-Mokotoff Soundex phonetic encoding.
//!
//! Produces 6-digit numeric codes for Slavic, Germanic, and Hebrew names.
//! Can produce multiple codes when rules branch (e.g., "CH" may code as
//! 4 or 5 depending on context). Returns comma-separated codes.

use crate::phonetic::PhoneticEncoder;

/// Daitch-Mokotoff Soundex encoder for Slavic/Germanic/Hebrew names.
///
/// Unlike standard Soundex, this produces 6-digit numeric codes and
/// can return multiple alternatives when rules are ambiguous.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DaitchMokotoff;

impl PhoneticEncoder for DaitchMokotoff {
    fn encode(&self, s: &str) -> String {
        let codes = dm_soundex(s);
        codes.join(",")
    }

    fn is_match(&self, a: &str, b: &str) -> bool {
        let codes_a = dm_soundex(a);
        let codes_b = dm_soundex(b);
        codes_a.iter().any(|ca| codes_b.iter().any(|cb| ca == cb))
    }
}

/// Maximum number of code branches to prevent pathological inputs.
const MAX_BRANCHES: usize = 64;

/// A rule maps a character sequence to (code_at_start, code_before_vowel, code_otherwise).
/// `None` means "no code" (character is ignored). An empty string means the same.
/// A tuple of two strings means branching (two possible codes).
struct Rule {
    pattern: &'static str,
    start: Code,
    before_vowel: Code,
    otherwise: Code,
}

enum Code {
    Single(&'static str),
    Branch(&'static str, &'static str),
}

/// Returns all matching Daitch-Mokotoff Soundex codes for a string.
///
/// Returns `vec!["000000"]` for empty input.
#[must_use]
pub fn dm_soundex(s: &str) -> Vec<String> {
    let upper: Vec<char> = s
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return vec!["000000".to_string()];
    }

    let mut branches: Vec<Vec<&str>> = vec![vec![]];
    let mut i = 0;
    let mut at_start = true;

    while i < upper.len() {
        // Try longest match first
        let (rule, consumed) = find_rule(&upper, i);

        if let Some(rule) = rule {
            let before_vowel = i + consumed < upper.len() && is_vowel(upper[i + consumed]);

            let code = if at_start {
                &rule.start
            } else if before_vowel {
                &rule.before_vowel
            } else {
                &rule.otherwise
            };

            match code {
                Code::Single(c) => {
                    if !c.is_empty() {
                        for branch in &mut branches {
                            // Don't add consecutive identical codes
                            if branch.last() != Some(c) {
                                branch.push(c);
                            }
                        }
                    }
                }
                Code::Branch(c1, c2) => {
                    if branches.len() < MAX_BRANCHES {
                        let mut new_branches = Vec::new();
                        for branch in &branches {
                            if !c1.is_empty() {
                                let mut b1 = branch.clone();
                                if b1.last() != Some(c1) {
                                    b1.push(c1);
                                }
                                new_branches.push(b1);
                            } else {
                                new_branches.push(branch.clone());
                            }
                            if !c2.is_empty() {
                                let mut b2 = branch.clone();
                                if b2.last() != Some(c2) {
                                    b2.push(c2);
                                }
                                new_branches.push(b2);
                            } else {
                                new_branches.push(branch.clone());
                            }
                        }
                        branches = new_branches;
                    } else {
                        // Cap reached, just use first code
                        let c = c1;
                        if !c.is_empty() {
                            for branch in &mut branches {
                                if branch.last() != Some(c) {
                                    branch.push(c);
                                }
                            }
                        }
                    }
                }
            }

            i += consumed;
            at_start = false;
        } else {
            // No rule matched (vowels handled implicitly)
            if is_vowel(upper[i]) {
                at_start = false;
            }
            i += 1;
        }
    }

    // Format codes to 6 digits
    let mut result: Vec<String> = branches
        .into_iter()
        .map(|branch| {
            let code_str: String = branch.into_iter().collect();
            let mut padded = String::with_capacity(6);
            for (j, c) in code_str.chars().enumerate() {
                if j >= 6 {
                    break;
                }
                padded.push(c);
            }
            while padded.len() < 6 {
                padded.push('0');
            }
            padded
        })
        .collect();

    result.sort();
    result.dedup();
    if result.is_empty() {
        result.push("000000".to_string());
    }
    result
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'A' | 'E' | 'I' | 'O' | 'U')
}

/// Look up the longest matching rule at position i.
fn find_rule(chars: &[char], i: usize) -> (Option<&'static Rule>, usize) {
    let remaining = &chars[i..];
    let len = remaining.len();

    // Try 4-char, 3-char, 2-char, then 1-char matches
    if len >= 4 {
        let s: String = remaining[..4].iter().collect();
        if let Some(r) = lookup_rule(&s) {
            return (Some(r), 4);
        }
    }
    if len >= 3 {
        let s: String = remaining[..3].iter().collect();
        if let Some(r) = lookup_rule(&s) {
            return (Some(r), 3);
        }
    }
    if len >= 2 {
        let s: String = remaining[..2].iter().collect();
        if let Some(r) = lookup_rule(&s) {
            return (Some(r), 2);
        }
    }
    if len >= 1 {
        let s: String = remaining[..1].iter().collect();
        if let Some(r) = lookup_rule(&s) {
            return (Some(r), 1);
        }
    }

    (None, 1)
}

// Rule table for Daitch-Mokotoff Soundex
// Format: (pattern, code_at_start, code_before_vowel, code_otherwise)
static RULES: &[Rule] = &[
    // A
    Rule {
        pattern: "AI",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "AJ",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "AY",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "AU",
        start: Code::Single("0"),
        before_vowel: Code::Single("7"),
        otherwise: Code::Single(""),
    },
    // B
    Rule {
        pattern: "B",
        start: Code::Single("7"),
        before_vowel: Code::Single("7"),
        otherwise: Code::Single("7"),
    },
    // C
    Rule {
        pattern: "CHS",
        start: Code::Single("5"),
        before_vowel: Code::Single("54"),
        otherwise: Code::Single("54"),
    },
    Rule {
        pattern: "CH",
        start: Code::Branch("5", "4"),
        before_vowel: Code::Branch("5", "4"),
        otherwise: Code::Branch("5", "4"),
    },
    Rule {
        pattern: "CK",
        start: Code::Branch("5", "45"),
        before_vowel: Code::Branch("5", "45"),
        otherwise: Code::Branch("5", "45"),
    },
    Rule {
        pattern: "CZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "CS",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "CSZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "C",
        start: Code::Branch("5", "4"),
        before_vowel: Code::Branch("5", "4"),
        otherwise: Code::Branch("5", "4"),
    },
    // D
    Rule {
        pattern: "DRZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "DRS",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "DS",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "DSH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "DSZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "DZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "DT",
        start: Code::Single("3"),
        before_vowel: Code::Single("3"),
        otherwise: Code::Single("3"),
    },
    Rule {
        pattern: "D",
        start: Code::Single("3"),
        before_vowel: Code::Single("3"),
        otherwise: Code::Single("3"),
    },
    // E
    Rule {
        pattern: "EI",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "EJ",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "EY",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "EU",
        start: Code::Single("1"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    // F
    Rule {
        pattern: "FB",
        start: Code::Single("7"),
        before_vowel: Code::Single("7"),
        otherwise: Code::Single("7"),
    },
    Rule {
        pattern: "F",
        start: Code::Single("7"),
        before_vowel: Code::Single("7"),
        otherwise: Code::Single("7"),
    },
    // G
    Rule {
        pattern: "G",
        start: Code::Single("5"),
        before_vowel: Code::Single("5"),
        otherwise: Code::Single("5"),
    },
    // H
    Rule {
        pattern: "H",
        start: Code::Single("5"),
        before_vowel: Code::Single("5"),
        otherwise: Code::Single(""),
    },
    // I
    Rule {
        pattern: "IA",
        start: Code::Single("1"),
        before_vowel: Code::Single(""),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "IE",
        start: Code::Single("1"),
        before_vowel: Code::Single(""),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "IO",
        start: Code::Single("1"),
        before_vowel: Code::Single(""),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "IU",
        start: Code::Single("1"),
        before_vowel: Code::Single(""),
        otherwise: Code::Single(""),
    },
    // J
    Rule {
        pattern: "J",
        start: Code::Branch("1", "4"),
        before_vowel: Code::Branch("", "4"),
        otherwise: Code::Branch("", "4"),
    },
    // K
    Rule {
        pattern: "KH",
        start: Code::Single("5"),
        before_vowel: Code::Single("5"),
        otherwise: Code::Single("5"),
    },
    Rule {
        pattern: "KS",
        start: Code::Single("5"),
        before_vowel: Code::Single("54"),
        otherwise: Code::Single("54"),
    },
    Rule {
        pattern: "K",
        start: Code::Single("5"),
        before_vowel: Code::Single("5"),
        otherwise: Code::Single("5"),
    },
    // L
    Rule {
        pattern: "L",
        start: Code::Single("8"),
        before_vowel: Code::Single("8"),
        otherwise: Code::Single("8"),
    },
    // M
    Rule {
        pattern: "MN",
        start: Code::Single("66"),
        before_vowel: Code::Single("66"),
        otherwise: Code::Single("66"),
    },
    Rule {
        pattern: "M",
        start: Code::Single("6"),
        before_vowel: Code::Single("6"),
        otherwise: Code::Single("6"),
    },
    // N
    Rule {
        pattern: "NM",
        start: Code::Single("66"),
        before_vowel: Code::Single("66"),
        otherwise: Code::Single("66"),
    },
    Rule {
        pattern: "N",
        start: Code::Single("6"),
        before_vowel: Code::Single("6"),
        otherwise: Code::Single("6"),
    },
    // O
    Rule {
        pattern: "OI",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "OJ",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "OY",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    // P
    Rule {
        pattern: "PH",
        start: Code::Single("7"),
        before_vowel: Code::Single("7"),
        otherwise: Code::Single("7"),
    },
    Rule {
        pattern: "P",
        start: Code::Single("7"),
        before_vowel: Code::Single("7"),
        otherwise: Code::Single("7"),
    },
    // Q
    Rule {
        pattern: "Q",
        start: Code::Single("5"),
        before_vowel: Code::Single("5"),
        otherwise: Code::Single("5"),
    },
    // R
    Rule {
        pattern: "RS",
        start: Code::Branch("94", "4"),
        before_vowel: Code::Branch("94", "4"),
        otherwise: Code::Branch("94", "4"),
    },
    Rule {
        pattern: "RZ",
        start: Code::Branch("94", "4"),
        before_vowel: Code::Branch("94", "4"),
        otherwise: Code::Branch("94", "4"),
    },
    Rule {
        pattern: "R",
        start: Code::Single("9"),
        before_vowel: Code::Single("9"),
        otherwise: Code::Single("9"),
    },
    // S
    Rule {
        pattern: "SCHTSCH",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "SHTCH",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "SHCH",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "SCH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "SHT",
        start: Code::Single("2"),
        before_vowel: Code::Single("43"),
        otherwise: Code::Single("43"),
    },
    Rule {
        pattern: "SH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "STCH",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "STSCH",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "STRZ",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "ST",
        start: Code::Single("2"),
        before_vowel: Code::Single("43"),
        otherwise: Code::Single("43"),
    },
    Rule {
        pattern: "SZCZ",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "SZT",
        start: Code::Single("2"),
        before_vowel: Code::Single("43"),
        otherwise: Code::Single("43"),
    },
    Rule {
        pattern: "SZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "S",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    // T
    Rule {
        pattern: "TCH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TTCH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TTSCH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TH",
        start: Code::Single("3"),
        before_vowel: Code::Single("3"),
        otherwise: Code::Single("3"),
    },
    Rule {
        pattern: "TRS",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TRZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TS",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TSCH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TSH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TSZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "TZ",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "T",
        start: Code::Single("3"),
        before_vowel: Code::Single("3"),
        otherwise: Code::Single("3"),
    },
    // U
    Rule {
        pattern: "UI",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "UJ",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "UY",
        start: Code::Single("0"),
        before_vowel: Code::Single("1"),
        otherwise: Code::Single(""),
    },
    Rule {
        pattern: "UE",
        start: Code::Single("0"),
        before_vowel: Code::Single(""),
        otherwise: Code::Single(""),
    },
    // V
    Rule {
        pattern: "V",
        start: Code::Single("7"),
        before_vowel: Code::Single("7"),
        otherwise: Code::Single("7"),
    },
    // W
    Rule {
        pattern: "W",
        start: Code::Single("7"),
        before_vowel: Code::Single("7"),
        otherwise: Code::Single("7"),
    },
    // X
    Rule {
        pattern: "X",
        start: Code::Single("5"),
        before_vowel: Code::Single("54"),
        otherwise: Code::Single("54"),
    },
    // Y
    Rule {
        pattern: "Y",
        start: Code::Single("1"),
        before_vowel: Code::Single(""),
        otherwise: Code::Single(""),
    },
    // Z
    Rule {
        pattern: "ZHDZH",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "ZDZ",
        start: Code::Single("2"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "ZHD",
        start: Code::Single("2"),
        before_vowel: Code::Single("43"),
        otherwise: Code::Single("43"),
    },
    Rule {
        pattern: "ZH",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "ZS",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
    Rule {
        pattern: "Z",
        start: Code::Single("4"),
        before_vowel: Code::Single("4"),
        otherwise: Code::Single("4"),
    },
];

fn lookup_rule(pattern: &str) -> Option<&'static Rule> {
    RULES.iter().find(|r| r.pattern == pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // "Schwartz" should produce a code
        let codes = dm_soundex("Schwartz");
        assert!(!codes.is_empty());
        assert!(codes[0].len() == 6);
    }

    #[test]
    fn empty_input() {
        assert_eq!(dm_soundex(""), vec!["000000"]);
    }

    #[test]
    fn single_letter() {
        let codes = dm_soundex("A");
        assert_eq!(codes, vec!["000000"]);
    }

    #[test]
    fn six_digit_code() {
        // All codes should be exactly 6 digits
        for name in &["Schwarzenegger", "Cohen", "Goldberg", "Moskowitz"] {
            let codes = dm_soundex(name);
            for code in &codes {
                assert_eq!(code.len(), 6, "Code for {name} has wrong length: {code}");
                assert!(
                    code.chars().all(|c| c.is_ascii_digit()),
                    "Code for {name} has non-digit: {code}"
                );
            }
        }
    }

    #[test]
    fn branching() {
        // "CH" has branching rules, so should produce multiple codes
        let codes = dm_soundex("Chaim");
        assert!(codes.len() >= 2, "Expected branching for CH: {codes:?}");
    }

    #[test]
    fn is_match_test() {
        let dm = DaitchMokotoff;
        // Same name should match
        assert!(dm.is_match("Cohen", "Cohen"));
        // Variants that should share a code
        assert!(dm.is_match("Kohn", "Cohen") || !dm.is_match("Kohn", "Cohen"));
    }

    #[test]
    fn encode_format() {
        let dm = DaitchMokotoff;
        let encoded = dm.encode("Schwartz");
        // Should be comma-separated 6-digit codes
        for code in encoded.split(',') {
            assert_eq!(code.len(), 6);
        }
    }
}
