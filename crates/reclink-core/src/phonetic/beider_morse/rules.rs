//! Phonetic transformation rules for Beider-Morse.
//!
//! Each rule is a `(pattern, context_before, context_after, replacement)` tuple.
//! Rules are applied left-to-right, consuming matched characters. When multiple
//! replacements are possible, all variants are generated (branching).

/// A single phonetic transformation rule.
#[derive(Debug, Clone)]
pub struct Rule {
    /// The input pattern to match.
    pub pattern: &'static str,
    /// Context required before the pattern (empty = any).
    pub context_before: &'static str,
    /// Context required after the pattern (empty = any).
    pub context_after: &'static str,
    /// The phonetic replacement(s). Multiple options separated by `|`.
    pub replacement: &'static str,
}

impl Rule {
    const fn new(
        pattern: &'static str,
        context_before: &'static str,
        context_after: &'static str,
        replacement: &'static str,
    ) -> Self {
        Self {
            pattern,
            context_before,
            context_after,
            replacement,
        }
    }
}

/// Check if context_before matches at the given position.
fn matches_before(input: &str, pos: usize, ctx: &str) -> bool {
    if ctx.is_empty() {
        return true;
    }
    if pos < ctx.len() {
        return false;
    }
    input[pos - ctx.len()..pos].eq_ignore_ascii_case(ctx)
}

/// Check if context_after matches at the given position.
fn matches_after(input: &str, pos: usize, ctx: &str) -> bool {
    if ctx.is_empty() {
        return true;
    }
    if pos + ctx.len() > input.len() {
        return false;
    }
    input[pos..pos + ctx.len()].eq_ignore_ascii_case(ctx)
}

/// Apply a rule set to an input string, returning all phonetic variants.
///
/// The algorithm walks through the input left-to-right. At each position,
/// it tries rules in order. The first matching rule's replacement is used.
/// If a replacement contains `|`, the computation branches into multiple
/// output variants. A maximum of `max_variants` outputs are produced.
pub fn apply_rules(input: &str, rules: &[Rule], max_variants: usize) -> Vec<String> {
    let lower = input.to_lowercase();
    let bytes = lower.as_str();
    let mut results = vec![String::new()];
    let mut pos = 0;

    while pos < bytes.len() {
        let mut matched = false;

        for rule in rules {
            let pat = rule.pattern;
            if pos + pat.len() > bytes.len() {
                continue;
            }
            if !bytes[pos..].starts_with(pat) {
                continue;
            }
            if !matches_before(bytes, pos, rule.context_before) {
                continue;
            }
            if !matches_after(bytes, pos + pat.len(), rule.context_after) {
                continue;
            }

            // Rule matched — apply replacement(s)
            let replacements: Vec<&str> = rule.replacement.split('|').collect();

            if replacements.len() == 1 {
                for result in &mut results {
                    result.push_str(replacements[0]);
                }
            } else {
                // Branch: multiply results by number of replacements
                let mut new_results = Vec::with_capacity(results.len() * replacements.len());
                for result in &results {
                    for &repl in &replacements {
                        if new_results.len() >= max_variants {
                            break;
                        }
                        let mut new = result.clone();
                        new.push_str(repl);
                        new_results.push(new);
                    }
                }
                results = new_results;
            }

            pos += pat.len();
            matched = true;
            break;
        }

        if !matched {
            // No rule matched — pass character through
            let ch = bytes[pos..].chars().next().unwrap();
            for result in &mut results {
                result.push(ch);
            }
            pos += ch.len_utf8();
        }
    }

    // Truncate to max_variants
    results.truncate(max_variants);
    results
}

// ─── Generic (language-agnostic) rules ────────────────────────────────────

/// Generic phonetic rules applicable to most Western European names.
pub static GENERIC_RULES: &[Rule] = &[
    // Silent letters and digraphs
    Rule::new("gn", "", "", "n|gn"),
    Rule::new("kn", "", "", "n"),
    Rule::new("pn", "", "", "n"),
    Rule::new("wr", "", "", "r"),
    Rule::new("mb", "", "", "m"),
    // Consonant clusters
    Rule::new("sch", "", "", "S|sk"),
    Rule::new("tch", "", "", "tS"),
    Rule::new("tsh", "", "", "tS"),
    Rule::new("tsch", "", "", "tS"),
    Rule::new("ch", "", "", "x|tS|S"),
    Rule::new("ck", "", "", "k"),
    Rule::new("sh", "", "", "S"),
    Rule::new("ph", "", "", "f"),
    Rule::new("th", "", "", "t"),
    Rule::new("gh", "", "", "|g"),
    Rule::new("wh", "", "", "w"),
    Rule::new("qu", "", "", "kw"),
    Rule::new("rh", "", "", "r"),
    Rule::new("dg", "", "", "dZ"),
    // Double consonants → single
    Rule::new("bb", "", "", "b"),
    Rule::new("cc", "", "", "k|ts"),
    Rule::new("dd", "", "", "d"),
    Rule::new("ff", "", "", "f"),
    Rule::new("gg", "", "", "g"),
    Rule::new("hh", "", "", ""),
    Rule::new("kk", "", "", "k"),
    Rule::new("ll", "", "", "l"),
    Rule::new("mm", "", "", "m"),
    Rule::new("nn", "", "", "n"),
    Rule::new("pp", "", "", "p"),
    Rule::new("rr", "", "", "r"),
    Rule::new("ss", "", "", "s"),
    Rule::new("tt", "", "", "t"),
    Rule::new("zz", "", "", "ts|z"),
    // Vowel combinations
    Rule::new("ae", "", "", "e|aj"),
    Rule::new("ai", "", "", "aj"),
    Rule::new("au", "", "", "aw"),
    Rule::new("ay", "", "", "aj"),
    Rule::new("ea", "", "", "e|ia"),
    Rule::new("ee", "", "", "i"),
    Rule::new("ei", "", "", "aj|ej"),
    Rule::new("eo", "", "", "io"),
    Rule::new("eu", "", "", "oj|ew"),
    Rule::new("ey", "", "", "aj|ej"),
    Rule::new("ie", "", "", "i|ie"),
    Rule::new("io", "", "", "io"),
    Rule::new("oa", "", "", "o"),
    Rule::new("oo", "", "", "u"),
    Rule::new("ou", "", "", "u|aw"),
    Rule::new("ow", "", "", "ow|aw"),
    Rule::new("oy", "", "", "oj"),
    Rule::new("ue", "", "", "u"),
    Rule::new("ui", "", "", "i|uj"),
    // Simple consonant mappings
    Rule::new("c", "", "e", "s|ts"),
    Rule::new("c", "", "i", "s|ts"),
    Rule::new("c", "", "y", "s"),
    Rule::new("c", "", "", "k"),
    Rule::new("g", "", "e", "dZ|g"),
    Rule::new("g", "", "i", "dZ|g"),
    Rule::new("g", "", "y", "dZ|g"),
    Rule::new("g", "", "", "g"),
    Rule::new("h", "", "", "|h"),
    Rule::new("j", "", "", "j|dZ|x"),
    Rule::new("q", "", "", "k"),
    Rule::new("s", "", "", "s|z"),
    Rule::new("w", "", "", "v|w"),
    Rule::new("x", "", "", "ks"),
    Rule::new("y", "", "", "i|j"),
    Rule::new("z", "", "", "z|ts"),
    // Pass-through vowels
    Rule::new("a", "", "", "a"),
    Rule::new("b", "", "", "b"),
    Rule::new("d", "", "", "d"),
    Rule::new("e", "", "", "e|"),
    Rule::new("f", "", "", "f"),
    Rule::new("i", "", "", "i"),
    Rule::new("k", "", "", "k"),
    Rule::new("l", "", "", "l"),
    Rule::new("m", "", "", "m"),
    Rule::new("n", "", "", "n"),
    Rule::new("o", "", "", "o"),
    Rule::new("p", "", "", "p"),
    Rule::new("r", "", "", "r"),
    Rule::new("t", "", "", "t"),
    Rule::new("u", "", "", "u"),
    Rule::new("v", "", "", "v"),
];

// ─── Ashkenazi-specific rules ─────────────────────────────────────────────

/// Rules specific to Ashkenazi (Yiddish/Hebrew-origin) names.
pub static ASHKENAZI_RULES: &[Rule] = &[
    // Yiddish-specific digraphs
    Rule::new("tz", "", "", "ts"),
    Rule::new("tsch", "", "", "tS"),
    Rule::new("sch", "", "", "S"),
    Rule::new("tch", "", "", "tS"),
    Rule::new("ch", "", "", "x"),
    Rule::new("sh", "", "", "S"),
    Rule::new("ph", "", "", "f"),
    Rule::new("th", "", "", "t"),
    Rule::new("gh", "", "", "g"),
    Rule::new("ck", "", "", "k"),
    Rule::new("wh", "", "", "v"),
    Rule::new("qu", "", "", "kv"),
    Rule::new("dg", "", "", "dZ"),
    // Yiddish vowel patterns
    Rule::new("ei", "", "", "aj"),
    Rule::new("ey", "", "", "aj"),
    Rule::new("ai", "", "", "aj"),
    Rule::new("ay", "", "", "aj"),
    Rule::new("au", "", "", "aw"),
    Rule::new("ou", "", "", "u"),
    Rule::new("oo", "", "", "u"),
    Rule::new("ee", "", "", "i"),
    Rule::new("ie", "", "", "i"),
    Rule::new("ae", "", "", "e"),
    Rule::new("ue", "", "", "i"),
    Rule::new("ui", "", "", "i"),
    Rule::new("oi", "", "", "oj"),
    Rule::new("oy", "", "", "oj"),
    // Double consonants
    Rule::new("bb", "", "", "b"),
    Rule::new("cc", "", "", "k"),
    Rule::new("dd", "", "", "d"),
    Rule::new("ff", "", "", "f"),
    Rule::new("gg", "", "", "g"),
    Rule::new("kk", "", "", "k"),
    Rule::new("ll", "", "", "l"),
    Rule::new("mm", "", "", "m"),
    Rule::new("nn", "", "", "n"),
    Rule::new("pp", "", "", "p"),
    Rule::new("rr", "", "", "r"),
    Rule::new("ss", "", "", "s"),
    Rule::new("tt", "", "", "t"),
    Rule::new("zz", "", "", "ts"),
    // Consonant context rules
    Rule::new("c", "", "e", "ts"),
    Rule::new("c", "", "i", "ts"),
    Rule::new("c", "", "y", "s"),
    Rule::new("c", "", "", "k"),
    Rule::new("g", "", "e", "g"),
    Rule::new("g", "", "i", "g"),
    Rule::new("g", "", "", "g"),
    Rule::new("h", "", "", "|h"),
    Rule::new("j", "", "", "j"),
    Rule::new("q", "", "", "k"),
    Rule::new("s", "", "", "s|z"),
    Rule::new("w", "", "", "v"),
    Rule::new("x", "", "", "ks"),
    Rule::new("y", "", "", "i|j"),
    Rule::new("z", "", "", "z|ts"),
    // Pass-through
    Rule::new("a", "", "", "a"),
    Rule::new("b", "", "", "b"),
    Rule::new("d", "", "", "d"),
    Rule::new("e", "", "", "e|"),
    Rule::new("f", "", "", "f"),
    Rule::new("i", "", "", "i"),
    Rule::new("k", "", "", "k"),
    Rule::new("l", "", "", "l"),
    Rule::new("m", "", "", "m"),
    Rule::new("n", "", "", "n"),
    Rule::new("o", "", "", "o"),
    Rule::new("p", "", "", "p"),
    Rule::new("r", "", "", "r"),
    Rule::new("t", "", "", "t"),
    Rule::new("u", "", "", "u"),
    Rule::new("v", "", "", "v"),
];

// ─── Final phonetic cleanup rules ─────────────────────────────────────────

/// Simplification rules applied after initial phonetic encoding.
/// Removes duplicate consecutive characters and normalizes.
pub fn cleanup(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut last = '\0';

    for ch in s.chars() {
        if ch != last {
            result.push(ch);
            last = ch;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_simple_rule() {
        let rules = &[
            Rule::new("sh", "", "", "S"),
            Rule::new("s", "", "", "s"),
            Rule::new("h", "", "", "h"),
            Rule::new("a", "", "", "a"),
            Rule::new("w", "", "", "w"),
        ];
        let result = apply_rules("shaw", rules, 10);
        assert_eq!(result, vec!["Saw"]);
    }

    #[test]
    fn branching_rules() {
        let rules = &[
            Rule::new("ch", "", "", "x|tS"),
            Rule::new("a", "", "", "a"),
            Rule::new("t", "", "", "t"),
        ];
        let result = apply_rules("chat", rules, 10);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"xat".to_string()));
        assert!(result.contains(&"tSat".to_string()));
    }

    #[test]
    fn context_before() {
        let rules = &[
            Rule::new("c", "s", "", ""),
            Rule::new("c", "", "", "k"),
            Rule::new("s", "", "", "s"),
            Rule::new("e", "", "", "e"),
        ];
        let result = apply_rules("sce", rules, 10);
        // "s" → "s", then "c" with context_before "s" → "", then "e" → "e"
        assert_eq!(result, vec!["se"]);
    }

    #[test]
    fn context_after() {
        let rules = &[
            Rule::new("c", "", "e", "s"),
            Rule::new("c", "", "", "k"),
            Rule::new("e", "", "", "e"),
        ];
        let result = apply_rules("ce", rules, 10);
        assert_eq!(result, vec!["se"]);
    }

    #[test]
    fn generic_rules_basic() {
        let result = apply_rules("smith", GENERIC_RULES, 10);
        assert!(!result.is_empty());
        // Should produce something like "smit" (th → t)
        assert!(result.iter().any(|r| r.contains('t')));
    }

    #[test]
    fn ashkenazi_rules_basic() {
        let result = apply_rules("schwartz", ASHKENAZI_RULES, 10);
        assert!(!result.is_empty());
    }

    #[test]
    fn cleanup_removes_duplicates() {
        assert_eq!(cleanup("aabbcc"), "abc");
        assert_eq!(cleanup("hello"), "helo");
        assert_eq!(cleanup(""), "");
    }

    #[test]
    fn max_variants_cap() {
        // Rules that would generate many variants
        let rules = &[
            Rule::new("a", "", "", "a|b|c"),
            Rule::new("b", "", "", "x|y|z"),
        ];
        let result = apply_rules("ab", rules, 4);
        assert!(result.len() <= 4);
    }
}
