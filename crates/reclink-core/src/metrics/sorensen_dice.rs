//! Sorensen-Dice coefficient on character bigrams.

use ahash::AHashMap;

use crate::metrics::SimilarityMetric;

/// Sorensen-Dice coefficient measures the overlap between two sets of
/// character bigrams. It is defined as 2 * |intersection| / (|A| + |B|).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SorensenDice;

impl SimilarityMetric for SorensenDice {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        sorensen_dice_similarity(a, b)
    }
}

/// Builds a bigram multiset (frequency map).
fn bigrams(s: &str) -> AHashMap<(char, char), u32> {
    let chars: Vec<char> = s.chars().collect();
    let mut freq = AHashMap::new();
    for pair in chars.windows(2) {
        *freq.entry((pair[0], pair[1])).or_insert(0) += 1;
    }
    freq
}

/// Computes the Sorensen-Dice coefficient between character bigrams of two strings.
#[must_use]
pub fn sorensen_dice_similarity(a: &str, b: &str) -> f64 {
    let bg_a = bigrams(a);
    let bg_b = bigrams(b);

    let total = bg_a.values().sum::<u32>() + bg_b.values().sum::<u32>();

    if total == 0 {
        return 1.0;
    }

    let mut intersection = 0u32;
    for (bigram, &count_a) in &bg_a {
        if let Some(&count_b) = bg_b.get(bigram) {
            intersection += count_a.min(count_b);
        }
    }

    (2 * intersection) as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical() {
        assert!(approx_eq(sorensen_dice_similarity("night", "night"), 1.0));
    }

    #[test]
    fn empty_strings() {
        assert!(approx_eq(sorensen_dice_similarity("", ""), 1.0));
        assert!(approx_eq(sorensen_dice_similarity("abc", ""), 0.0));
    }

    #[test]
    fn known_values() {
        // "night" bigrams: ni, ig, gh, ht
        // "nacht" bigrams: na, ac, ch, ht
        // intersection: ht (1)
        // dice = 2*1 / (4+4) = 0.25
        assert!(approx_eq(sorensen_dice_similarity("night", "nacht"), 0.25));
    }

    #[test]
    fn symmetry() {
        let a = sorensen_dice_similarity("abc", "bcd");
        let b = sorensen_dice_similarity("bcd", "abc");
        assert!(approx_eq(a, b));
    }

    #[test]
    fn single_char() {
        // Single chars produce no bigrams, so both-empty returns 1.0
        assert!(approx_eq(sorensen_dice_similarity("a", "a"), 1.0));
        assert!(approx_eq(sorensen_dice_similarity("a", "b"), 1.0));
    }
}
