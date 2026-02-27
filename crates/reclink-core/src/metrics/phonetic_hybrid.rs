//! Phonetic + edit distance hybrid scorer.

use crate::metrics::{Metric, SimilarityMetric};
use crate::phonetic::{PhoneticAlgorithm, PhoneticEncoder};

/// Combines a phonetic match with an edit distance metric for a weighted
/// hybrid similarity score.
///
/// `score = phonetic_weight * phonetic_match + (1 - phonetic_weight) * edit_similarity`
///
/// where `phonetic_match` is 1.0 if phonetic codes match, 0.0 otherwise.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhoneticHybrid {
    /// The phonetic algorithm to use.
    pub phonetic_algorithm: PhoneticAlgorithm,
    /// The edit distance metric to use (boxed to break recursive type cycle).
    pub edit_metric: Box<Metric>,
    /// Weight for the phonetic component (0.0 to 1.0).
    pub phonetic_weight: f64,
}

impl Default for PhoneticHybrid {
    fn default() -> Self {
        use crate::metrics::JaroWinkler;
        use crate::phonetic::Soundex;

        Self {
            phonetic_algorithm: PhoneticAlgorithm::Soundex(Soundex),
            edit_metric: Box::new(Metric::JaroWinkler(JaroWinkler::default())),
            phonetic_weight: 0.3,
        }
    }
}

impl SimilarityMetric for PhoneticHybrid {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        let phonetic_match = if self.phonetic_algorithm.is_match(a, b) {
            1.0
        } else {
            0.0
        };
        let edit_sim = self.edit_metric.similarity(a, b);
        self.phonetic_weight * phonetic_match + (1.0 - self.phonetic_weight) * edit_sim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn identical_strings() {
        let m = PhoneticHybrid::default();
        assert!(approx_eq(m.similarity("hello", "hello"), 1.0));
    }

    #[test]
    fn phonetic_match_boost() {
        let m = PhoneticHybrid::default();
        // "Smith" and "Smyth" have the same Soundex code
        let sim_match = m.similarity("Smith", "Smyth");
        // Without phonetic weight, pure edit distance would be lower
        let edit_only = m.edit_metric.similarity("Smith", "Smyth");
        assert!(sim_match > edit_only);
    }

    #[test]
    fn no_phonetic_match() {
        let m = PhoneticHybrid::default();
        // "Smith" and "Jones" have different Soundex codes
        let sim = m.similarity("Smith", "Jones");
        // Score should be purely from edit distance (weighted down)
        let edit_sim = m.edit_metric.similarity("Smith", "Jones");
        assert!(approx_eq(sim, 0.7 * edit_sim));
    }

    #[test]
    fn empty_strings() {
        let m = PhoneticHybrid::default();
        // Both empty: phonetic codes match ("0000" == "0000"), edit sim = 1.0
        assert!(approx_eq(m.similarity("", ""), 1.0));
    }
}
