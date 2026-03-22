//! Explain mode: run multiple metrics on a string pair and return the breakdown.

use crate::metrics::{metric_from_name, Metric};

/// A single algorithm's contribution to the comparison.
#[derive(Debug, Clone)]
pub struct AlgorithmScore {
    /// Name of the algorithm.
    pub algorithm: String,
    /// Similarity score in [0, 1].
    pub score: f64,
}

/// Full explanation of a string comparison.
#[derive(Debug, Clone)]
pub struct ExplainResult {
    /// First string.
    pub a: String,
    /// Second string.
    pub b: String,
    /// Per-algorithm scores.
    pub scores: Vec<AlgorithmScore>,
}

/// All available metric names for the explain function.
const ALL_METRICS: &[&str] = &[
    "levenshtein",
    "damerau_levenshtein",
    "hamming",
    "jaro",
    "jaro_winkler",
    "cosine",
    "jaccard",
    "sorensen_dice",
    "weighted_levenshtein",
    "token_sort",
    "token_set",
    "partial_ratio",
    "lcs",
    "longest_common_substring",
    "ngram_similarity",
    "smith_waterman",
    "phonetic_hybrid",
    "ratcliff_obershelp",
    "needleman_wunsch",
    "gotoh",
    "monge_elkan",
];

/// Compare two strings with all available algorithms and return the breakdown.
#[must_use]
pub fn explain(a: &str, b: &str) -> ExplainResult {
    let scores = ALL_METRICS
        .iter()
        .filter_map(|name| {
            metric_from_name(name).ok().map(|m| AlgorithmScore {
                algorithm: (*name).to_string(),
                score: m.similarity(a, b),
            })
        })
        .collect();

    ExplainResult {
        a: a.to_string(),
        b: b.to_string(),
        scores,
    }
}

/// Compare two strings with a specific set of algorithms.
#[must_use]
pub fn explain_with(a: &str, b: &str, algorithms: &[Metric]) -> ExplainResult {
    let scores = algorithms
        .iter()
        .enumerate()
        .map(|(i, m)| AlgorithmScore {
            algorithm: metric_name(m).unwrap_or_else(|| format!("metric_{i}")),
            score: m.similarity(a, b),
        })
        .collect();

    ExplainResult {
        a: a.to_string(),
        b: b.to_string(),
        scores,
    }
}

fn metric_name(m: &Metric) -> Option<String> {
    let name = match m {
        Metric::Levenshtein(_) => "levenshtein",
        Metric::DamerauLevenshtein(_) => "damerau_levenshtein",
        Metric::Hamming(_) => "hamming",
        Metric::Jaro(_) => "jaro",
        Metric::JaroWinkler(_) => "jaro_winkler",
        Metric::Cosine(_) => "cosine",
        Metric::Jaccard(_) => "jaccard",
        Metric::SorensenDice(_) => "sorensen_dice",
        Metric::WeightedLevenshtein(_) => "weighted_levenshtein",
        Metric::TokenSort(_) => "token_sort",
        Metric::TokenSet(_) => "token_set",
        Metric::PartialRatio(_) => "partial_ratio",
        Metric::Lcs(_) => "lcs",
        Metric::LongestCommonSubstring(_) => "longest_common_substring",
        Metric::NgramSimilarity(_) => "ngram_similarity",
        Metric::SmithWaterman(_) => "smith_waterman",
        Metric::PhoneticHybrid(_) => "phonetic_hybrid",
        Metric::RatcliffObershelp(_) => "ratcliff_obershelp",
        Metric::NeedlemanWunsch(_) => "needleman_wunsch",
        Metric::Gotoh(_) => "gotoh",
        Metric::MongeElkan(_) => "monge_elkan",
        Metric::Custom { name, .. } => return Some(name.clone()),
    };
    Some(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explain_all() {
        let result = explain("kitten", "sitting");
        // Should have all 17 metrics (hamming may fail for different lengths,
        // but metric_from_name returns Ok and similarity() handles errors)
        assert!(!result.scores.is_empty());
        assert_eq!(result.a, "kitten");
        assert_eq!(result.b, "sitting");
        // Check that known metrics are present
        let names: Vec<&str> = result.scores.iter().map(|s| s.algorithm.as_str()).collect();
        assert!(names.contains(&"levenshtein"));
        assert!(names.contains(&"jaro_winkler"));
        assert!(names.contains(&"cosine"));
    }

    #[test]
    fn explain_with_specific() {
        let metrics = vec![
            metric_from_name("jaro").unwrap(),
            metric_from_name("cosine").unwrap(),
        ];
        let result = explain_with("hello", "world", &metrics);
        assert_eq!(result.scores.len(), 2);
        assert_eq!(result.scores[0].algorithm, "jaro");
        assert_eq!(result.scores[1].algorithm, "cosine");
    }

    #[test]
    fn explain_identical() {
        let result = explain("hello", "hello");
        for score in &result.scores {
            // All metrics should return ~1.0 for identical strings
            // (hamming may vary since it handles equal-length only)
            if score.algorithm != "hamming" {
                assert!(
                    score.score > 0.9,
                    "{} returned {} for identical strings",
                    score.algorithm,
                    score.score
                );
            }
        }
    }
}
