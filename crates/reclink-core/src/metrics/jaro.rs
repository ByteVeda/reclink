//! Jaro similarity metric.

use crate::metrics::SimilarityMetric;

/// Jaro similarity measures the edit distance between two strings,
/// accounting for character matches within a sliding window and transpositions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Jaro;

impl SimilarityMetric for Jaro {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        jaro_similarity(a, b)
    }
}

/// Computes the Jaro similarity between two strings.
///
/// Returns a value in \[0, 1\] where 1 means the strings are identical.
#[must_use]
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 && b_len == 0 {
        return 1.0;
    }
    if a_len == 0 || b_len == 0 {
        return 0.0;
    }

    let match_window = (a_len.max(b_len) / 2).saturating_sub(1);

    let mut a_matched = vec![false; a_len];
    let mut b_matched = vec![false; b_len];

    let mut matches = 0usize;
    for i in 0..a_len {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(b_len);

        for j in start..end {
            if !b_matched[j] && a_chars[i] == b_chars[j] {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut transpositions = 0usize;
    let mut k = 0;
    for i in 0..a_len {
        if !a_matched[i] {
            continue;
        }
        while !b_matched[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    (m / a_len as f64 + m / b_len as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(jaro_similarity("hello", "hello"), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(jaro_similarity("", ""), 1.0));
        assert!(approx_eq(jaro_similarity("abc", ""), 0.0));
        assert!(approx_eq(jaro_similarity("", "abc"), 0.0));
    }

    #[test]
    fn known_values() {
        assert!(approx_eq(jaro_similarity("martha", "marhta"), 0.9444));
        assert!(approx_eq(jaro_similarity("dwayne", "duane"), 0.8222));
        assert!(approx_eq(jaro_similarity("dixon", "dicksonx"), 0.7667));
    }

    #[test]
    fn symmetry() {
        let a = jaro_similarity("abc", "bac");
        let b = jaro_similarity("bac", "abc");
        assert!(approx_eq(a, b));
    }

    #[test]
    fn completely_different() {
        assert!(approx_eq(jaro_similarity("abc", "xyz"), 0.0));
    }
}
