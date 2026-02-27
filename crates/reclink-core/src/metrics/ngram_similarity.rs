//! N-gram similarity using Jaccard coefficient over character n-gram sets.

use ahash::AHashSet;

use crate::metrics::SimilarityMetric;

/// N-gram similarity computes the Jaccard coefficient over character n-gram
/// **sets** (not multisets). This differs from existing token-level Jaccard
/// (which uses words) and Sorensen-Dice (which uses multiset bigrams with Dice
/// formula).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NgramSimilarity {
    /// Size of character n-grams. Default: 2 (bigrams).
    pub n: usize,
}

impl Default for NgramSimilarity {
    fn default() -> Self {
        Self { n: 2 }
    }
}

impl SimilarityMetric for NgramSimilarity {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        ngram_similarity(a, b, self.n)
    }
}

/// Computes n-gram Jaccard similarity between two strings.
///
/// `|intersection(ngrams_a, ngrams_b)| / |union(ngrams_a, ngrams_b)|`
///
/// Returns 1.0 if both produce empty n-gram sets.
#[must_use]
pub fn ngram_similarity(a: &str, b: &str, n: usize) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let ngrams_a: AHashSet<&[char]> = if a_chars.len() >= n {
        a_chars.windows(n).collect()
    } else {
        AHashSet::new()
    };

    let ngrams_b: AHashSet<&[char]> = if b_chars.len() >= n {
        b_chars.windows(n).collect()
    } else {
        AHashSet::new()
    };

    if ngrams_a.is_empty() && ngrams_b.is_empty() {
        return 1.0;
    }
    if ngrams_a.is_empty() || ngrams_b.is_empty() {
        return 0.0;
    }

    let intersection = ngrams_a.intersection(&ngrams_b).count();
    let union = ngrams_a.union(&ngrams_b).count();

    intersection as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(ngram_similarity("hello", "hello", 2), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(ngram_similarity("", "", 2), 1.0));
        assert!(approx_eq(ngram_similarity("abc", "", 2), 0.0));
    }

    #[test]
    fn known_values() {
        // "night" bigrams: {ni, ig, gh, ht}
        // "nacht" bigrams: {na, ac, ch, ht}
        // intersection: {ht} = 1
        // union: {ni, ig, gh, ht, na, ac, ch} = 7
        // jaccard = 1/7
        assert!(approx_eq(ngram_similarity("night", "nacht", 2), 1.0 / 7.0));
    }

    #[test]
    fn configurable_n() {
        // With n=1 (unigrams), "abc" and "bcd" share {b, c}
        // union = {a, b, c, d}
        // jaccard = 2/4 = 0.5
        assert!(approx_eq(ngram_similarity("abc", "bcd", 1), 0.5));
    }

    #[test]
    fn short_strings() {
        // Single char produces no bigrams -> both empty -> 1.0
        assert!(approx_eq(ngram_similarity("a", "a", 2), 1.0));
        assert!(approx_eq(ngram_similarity("a", "b", 2), 1.0));
    }

    #[test]
    fn trigrams() {
        // "abcd" trigrams: {abc, bcd}
        // "bcde" trigrams: {bcd, cde}
        // intersection: {bcd} = 1, union = 3
        assert!(approx_eq(ngram_similarity("abcd", "bcde", 3), 1.0 / 3.0));
    }
}
