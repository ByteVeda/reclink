//! Cosine similarity on character n-grams.

use ahash::AHashMap;

use crate::metrics::SimilarityMetric;

/// Cosine similarity computes the cosine of the angle between two n-gram
/// frequency vectors derived from the input strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cosine {
    /// Size of character n-grams. Default: 2 (bigrams).
    pub n: usize,
}

impl Default for Cosine {
    fn default() -> Self {
        Self { n: 2 }
    }
}

impl SimilarityMetric for Cosine {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        cosine_similarity(a, b, self.n)
    }
}

/// Builds a character n-gram frequency map.
fn ngram_freq(s: &str, n: usize) -> AHashMap<Vec<char>, u32> {
    let chars: Vec<char> = s.chars().collect();
    let mut freq = AHashMap::new();
    if chars.len() < n {
        return freq;
    }
    for window in chars.windows(n) {
        *freq.entry(window.to_vec()).or_insert(0) += 1;
    }
    freq
}

/// Computes cosine similarity between character n-gram vectors of two strings.
#[must_use]
pub fn cosine_similarity(a: &str, b: &str, n: usize) -> f64 {
    let freq_a = ngram_freq(a, n);
    let freq_b = ngram_freq(b, n);

    if freq_a.is_empty() && freq_b.is_empty() {
        return 1.0;
    }
    if freq_a.is_empty() || freq_b.is_empty() {
        return 0.0;
    }

    let mut dot = 0u64;
    let mut norm_a = 0u64;
    let mut norm_b = 0u64;

    for (ngram, &count_a) in &freq_a {
        norm_a += (count_a as u64) * (count_a as u64);
        if let Some(&count_b) = freq_b.get(ngram) {
            dot += (count_a as u64) * (count_b as u64);
        }
    }
    for &count_b in freq_b.values() {
        norm_b += (count_b as u64) * (count_b as u64);
    }

    if norm_a == 0 || norm_b == 0 {
        return 0.0;
    }

    dot as f64 / ((norm_a as f64).sqrt() * (norm_b as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(cosine_similarity("hello", "hello", 2), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(cosine_similarity("", "", 2), 1.0));
        assert!(approx_eq(cosine_similarity("abc", "", 2), 0.0));
    }

    #[test]
    fn completely_different() {
        assert!(approx_eq(cosine_similarity("ab", "cd", 2), 0.0));
    }

    #[test]
    fn partial_overlap() {
        let sim = cosine_similarity("night", "nacht", 2);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn symmetry() {
        let a = cosine_similarity("abc", "bcd", 2);
        let b = cosine_similarity("bcd", "abc", 2);
        assert!(approx_eq(a, b));
    }

    #[test]
    fn short_strings() {
        // Single chars produce no bigrams, so both-empty returns 1.0
        assert!(approx_eq(cosine_similarity("a", "a", 2), 1.0));
        // Both "a" and "b" produce empty bigram sets -> treated as equal (both empty)
        assert!(approx_eq(cosine_similarity("a", "b", 2), 1.0));
    }
}
