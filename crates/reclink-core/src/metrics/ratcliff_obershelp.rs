//! Ratcliff-Obershelp (Gestalt Pattern Matching) similarity.
//!
//! Recursively finds the longest common substring, then recursively matches
//! the remaining left and right portions. Similarity is computed as
//! `2 * total_matching_chars / (len_a + len_b)`.

use crate::metrics::SimilarityMetric;

/// Ratcliff-Obershelp similarity (also known as Gestalt Pattern Matching).
///
/// Computes similarity by recursively finding the longest common substring
/// between two strings, then recursively matching the remaining portions
/// on either side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RatcliffObershelp;

impl SimilarityMetric for RatcliffObershelp {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        ratcliff_obershelp_similarity(a, b)
    }
}

/// Finds the longest common substring between two char slices.
/// Returns `(start_a, start_b, length)`.
fn longest_common_substr(a: &[char], b: &[char]) -> (usize, usize, usize) {
    let mut best_len = 0;
    let mut best_a = 0;
    let mut best_b = 0;

    // Single-row DP for longest common substring
    let mut prev = vec![0usize; b.len() + 1];

    for (i, ac) in a.iter().enumerate() {
        let mut prev_diag = 0;
        for (j, bc) in b.iter().enumerate() {
            let old = prev[j + 1];
            if ac == bc {
                prev[j + 1] = prev_diag + 1;
                if prev[j + 1] > best_len {
                    best_len = prev[j + 1];
                    best_a = i + 1 - best_len;
                    best_b = j + 1 - best_len;
                }
            } else {
                prev[j + 1] = 0;
            }
            prev_diag = old;
        }
    }

    (best_a, best_b, best_len)
}

/// Recursively counts matching characters using the Ratcliff-Obershelp algorithm.
fn matching_chars(a: &[char], b: &[char]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let (start_a, start_b, length) = longest_common_substr(a, b);
    if length == 0 {
        return 0;
    }

    // Count matching chars in left portions + this match + right portions
    let left = matching_chars(&a[..start_a], &b[..start_b]);
    let right = matching_chars(&a[start_a + length..], &b[start_b + length..]);

    left + length + right
}

/// Computes the Ratcliff-Obershelp similarity between two strings.
///
/// Returns a value in \[0, 1\] where 1 means identical.
/// Both empty strings are considered identical (returns 1.0).
#[must_use]
pub fn ratcliff_obershelp_similarity(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let total = a_chars.len() + b_chars.len();

    if total == 0 {
        return 1.0;
    }

    let matches = matching_chars(&a_chars, &b_chars);
    (2 * matches) as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(
            ratcliff_obershelp_similarity("hello", "hello"),
            1.0
        ));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(ratcliff_obershelp_similarity("", ""), 1.0));
        assert!(approx_eq(ratcliff_obershelp_similarity("abc", ""), 0.0));
        assert!(approx_eq(ratcliff_obershelp_similarity("", "abc"), 0.0));
    }

    #[test]
    fn completely_different() {
        assert!(approx_eq(ratcliff_obershelp_similarity("abc", "xyz"), 0.0));
    }

    #[test]
    fn partial_overlap() {
        let sim = ratcliff_obershelp_similarity("night", "nacht");
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn symmetry() {
        let a = ratcliff_obershelp_similarity("abc", "bcd");
        let b = ratcliff_obershelp_similarity("bcd", "abc");
        assert!(approx_eq(a, b));
    }

    #[test]
    fn known_value_python_difflib() {
        // Python difflib.SequenceMatcher("abcde", "abdce").ratio() ≈ 0.8
        let sim = ratcliff_obershelp_similarity("abcde", "abdce");
        assert!(approx_eq(sim, 0.8));
    }

    #[test]
    fn single_char() {
        assert!(approx_eq(ratcliff_obershelp_similarity("a", "a"), 1.0));
        assert!(approx_eq(ratcliff_obershelp_similarity("a", "b"), 0.0));
    }
}
