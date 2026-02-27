//! String similarity and distance metrics.
//!
//! This module provides 17 string comparison algorithms exposed through two traits:
//! - [`DistanceMetric`] for edit-distance style metrics (returning `usize`)
//! - [`SimilarityMetric`] for normalized similarity scores (returning `f64` in \[0, 1\])
//!
//! All algorithms are implemented from scratch with buffer reuse for batch operations.

pub mod alignment;
pub mod batch;
pub mod composite;
pub mod cosine;
pub mod damerau_levenshtein;
pub mod explain;
pub mod hamming;
pub mod jaccard;
pub mod jaro;
pub mod jaro_winkler;
pub mod lcs;
pub mod levenshtein;
pub mod longest_common_substring;
pub mod ngram_similarity;
pub mod partial_ratio;
pub mod phonetic_hybrid;
pub mod smith_waterman;
pub mod sorensen_dice;
pub mod streaming;
pub mod tfidf;
pub mod token_set;
pub mod token_sort;
pub mod weighted_levenshtein;

pub use batch::{match_batch, match_best, MatchResult};
pub use composite::CompositeScorer;
pub use cosine::Cosine;
pub use damerau_levenshtein::DamerauLevenshtein;
pub use hamming::Hamming;
pub use jaccard::Jaccard;
pub use jaro::Jaro;
pub use jaro_winkler::JaroWinkler;
pub use lcs::Lcs;
pub use levenshtein::Levenshtein;
pub use longest_common_substring::LongestCommonSubstring;
pub use ngram_similarity::NgramSimilarity;
pub use partial_ratio::PartialRatio;
pub use phonetic_hybrid::PhoneticHybrid;
pub use smith_waterman::SmithWaterman;
pub use sorensen_dice::SorensenDice;
pub use streaming::StreamingMatcher;
pub use tfidf::TfIdfMatcher;
pub use token_set::TokenSet;
pub use token_sort::TokenSort;
pub use weighted_levenshtein::WeightedLevenshtein;

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{ReclinkError, Result};

/// Default maximum string length (in characters) for metric computation.
/// Strings longer than this limit return 0.0 similarity / max distance.
const DEFAULT_MAX_STRING_LENGTH: usize = 10_000;

/// Global maximum string length. Use [`set_max_string_length`] and
/// [`get_max_string_length`] to configure at runtime.
static MAX_STRING_LENGTH: AtomicUsize = AtomicUsize::new(DEFAULT_MAX_STRING_LENGTH);

/// Set the maximum string length (in characters) for metric computation.
///
/// Strings exceeding this length will return 0.0 similarity without computing
/// the full algorithm, preventing OOM on adversarial or accidental huge inputs.
///
/// Set to `0` to disable the check entirely.
pub fn set_max_string_length(max_len: usize) {
    MAX_STRING_LENGTH.store(max_len, Ordering::Relaxed);
}

/// Get the current maximum string length limit.
#[must_use]
pub fn get_max_string_length() -> usize {
    MAX_STRING_LENGTH.load(Ordering::Relaxed)
}

/// Returns `true` if either string exceeds the configured max length.
/// A max length of 0 disables the check.
#[inline]
fn exceeds_max_length(a: &str, b: &str) -> bool {
    let max_len = MAX_STRING_LENGTH.load(Ordering::Relaxed);
    if max_len == 0 {
        return false;
    }
    a.chars().count() > max_len || b.chars().count() > max_len
}

/// A metric that computes edit distance between two strings.
///
/// Distance metrics return a non-negative integer representing the minimum
/// number of operations to transform one string into another.
pub trait DistanceMetric {
    /// Computes the raw distance between two strings.
    fn distance(&self, a: &str, b: &str) -> Result<usize>;

    /// Returns normalized distance in \[0, 1\], where 0 means identical.
    fn normalized_distance(&self, a: &str, b: &str) -> Result<f64> {
        let d = self.distance(a, b)?;
        let max_len = a.chars().count().max(b.chars().count());
        if max_len == 0 {
            return Ok(0.0);
        }
        Ok(d as f64 / max_len as f64)
    }

    /// Returns normalized similarity in \[0, 1\], where 1 means identical.
    fn normalized_similarity(&self, a: &str, b: &str) -> Result<f64> {
        Ok(1.0 - self.normalized_distance(a, b)?)
    }
}

/// A metric that computes similarity between two strings.
///
/// Similarity metrics return a `f64` in \[0, 1\] where 1 means identical.
pub trait SimilarityMetric {
    /// Computes the similarity between two strings, returning a value in \[0, 1\].
    fn similarity(&self, a: &str, b: &str) -> f64;

    /// Returns dissimilarity (1 - similarity).
    fn dissimilarity(&self, a: &str, b: &str) -> f64 {
        1.0 - self.similarity(a, b)
    }
}

/// Enum dispatch for all available metrics, avoiding vtable overhead in hot loops.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Levenshtein edit distance.
    Levenshtein(Levenshtein),
    /// Damerau-Levenshtein edit distance.
    DamerauLevenshtein(DamerauLevenshtein),
    /// Hamming distance.
    Hamming(Hamming),
    /// Jaro similarity.
    Jaro(Jaro),
    /// Jaro-Winkler similarity.
    JaroWinkler(JaroWinkler),
    /// Cosine similarity on character n-grams.
    Cosine(Cosine),
    /// Jaccard similarity on tokens.
    Jaccard(Jaccard),
    /// Sorensen-Dice coefficient.
    SorensenDice(SorensenDice),
    /// Weighted Levenshtein with configurable costs.
    WeightedLevenshtein(WeightedLevenshtein),
    /// Token Sort Ratio.
    TokenSort(TokenSort),
    /// Token Set Ratio.
    TokenSet(TokenSet),
    /// Partial Ratio (best substring match).
    PartialRatio(PartialRatio),
    /// Longest Common Subsequence similarity.
    Lcs(Lcs),
    /// Longest Common Substring similarity.
    LongestCommonSubstring(LongestCommonSubstring),
    /// N-gram Jaccard similarity.
    NgramSimilarity(NgramSimilarity),
    /// Smith-Waterman local alignment similarity.
    SmithWaterman(SmithWaterman),
    /// Phonetic + edit distance hybrid.
    PhoneticHybrid(PhoneticHybrid),
}

impl Metric {
    /// Computes a normalized similarity score in \[0, 1\] regardless of underlying metric type.
    ///
    /// Returns 0.0 immediately if either string exceeds the configured
    /// [`MAX_STRING_LENGTH`] (see [`set_max_string_length`]).
    #[must_use]
    pub fn similarity(&self, a: &str, b: &str) -> f64 {
        if exceeds_max_length(a, b) {
            return 0.0;
        }
        match self {
            Metric::Levenshtein(m) => m.normalized_similarity(a, b).unwrap_or(0.0),
            Metric::DamerauLevenshtein(m) => m.normalized_similarity(a, b).unwrap_or(0.0),
            Metric::Hamming(m) => m.normalized_similarity(a, b).unwrap_or(0.0),
            Metric::Jaro(m) => m.similarity(a, b),
            Metric::JaroWinkler(m) => m.similarity(a, b),
            Metric::Cosine(m) => m.similarity(a, b),
            Metric::Jaccard(m) => m.similarity(a, b),
            Metric::SorensenDice(m) => m.similarity(a, b),
            Metric::WeightedLevenshtein(m) => m.similarity(a, b),
            Metric::TokenSort(m) => m.similarity(a, b),
            Metric::TokenSet(m) => m.similarity(a, b),
            Metric::PartialRatio(m) => m.similarity(a, b),
            Metric::Lcs(m) => m.similarity(a, b),
            Metric::LongestCommonSubstring(m) => m.similarity(a, b),
            Metric::NgramSimilarity(m) => m.similarity(a, b),
            Metric::SmithWaterman(m) => m.similarity(a, b),
            Metric::PhoneticHybrid(m) => m.similarity(a, b),
        }
    }
}

impl Default for Metric {
    fn default() -> Self {
        Metric::JaroWinkler(JaroWinkler::default())
    }
}

/// Parses a metric name string into a [`Metric`] enum variant.
///
/// Supported names: `"levenshtein"`, `"damerau_levenshtein"`, `"hamming"`,
/// `"jaro"`, `"jaro_winkler"`, `"cosine"`, `"jaccard"`, `"sorensen_dice"`,
/// `"weighted_levenshtein"`, `"token_sort"`, `"token_set"`, `"partial_ratio"`,
/// `"lcs"`, `"longest_common_substring"`, `"ngram_similarity"`, `"smith_waterman"`,
/// `"phonetic_hybrid"`.
pub fn metric_from_name(name: &str) -> Result<Metric> {
    match name {
        "levenshtein" => Ok(Metric::Levenshtein(Levenshtein)),
        "damerau_levenshtein" => Ok(Metric::DamerauLevenshtein(DamerauLevenshtein)),
        "hamming" => Ok(Metric::Hamming(Hamming)),
        "jaro" => Ok(Metric::Jaro(Jaro)),
        "jaro_winkler" => Ok(Metric::JaroWinkler(JaroWinkler::default())),
        "cosine" => Ok(Metric::Cosine(Cosine::default())),
        "jaccard" => Ok(Metric::Jaccard(Jaccard)),
        "sorensen_dice" => Ok(Metric::SorensenDice(SorensenDice)),
        "weighted_levenshtein" => Ok(Metric::WeightedLevenshtein(WeightedLevenshtein::default())),
        "token_sort" => Ok(Metric::TokenSort(TokenSort)),
        "token_set" => Ok(Metric::TokenSet(TokenSet)),
        "partial_ratio" => Ok(Metric::PartialRatio(PartialRatio)),
        "lcs" => Ok(Metric::Lcs(Lcs)),
        "longest_common_substring" => Ok(Metric::LongestCommonSubstring(LongestCommonSubstring)),
        "ngram_similarity" => Ok(Metric::NgramSimilarity(NgramSimilarity::default())),
        "smith_waterman" => Ok(Metric::SmithWaterman(SmithWaterman::default())),
        "phonetic_hybrid" => Ok(Metric::PhoneticHybrid(PhoneticHybrid::default())),
        _ => Err(ReclinkError::InvalidConfig(format!(
            "unknown metric: `{name}`. Expected one of: levenshtein, damerau_levenshtein, \
             hamming, jaro, jaro_winkler, cosine, jaccard, sorensen_dice, \
             weighted_levenshtein, token_sort, token_set, partial_ratio, lcs, \
             longest_common_substring, ngram_similarity, smith_waterman, phonetic_hybrid"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // All max-string-length tests are in one function to avoid global-state
    // race conditions when cargo runs tests in parallel.
    #[test]
    fn max_string_length_behavior() {
        let original = get_max_string_length();

        // 1. Enforced: a small limit blocks large strings
        set_max_string_length(5);
        assert!(exceeds_max_length("abcdef", "ab")); // 6 > 5
        assert!(!exceeds_max_length("abc", "ab")); // 3 <= 5

        // 2. Similarity returns 0.0 for oversized strings
        let metric = Metric::JaroWinkler(JaroWinkler::default());
        assert_eq!(metric.similarity("abcdef", "abcdef"), 0.0);
        assert!(metric.similarity("abc", "abc") > 0.0);

        // 3. Roundtrip
        set_max_string_length(42);
        assert_eq!(get_max_string_length(), 42);

        // 4. Disabled with 0 — no string is "too long"
        set_max_string_length(0);
        assert!(!exceeds_max_length("a".repeat(100_000).as_str(), "b"));

        // Restore
        set_max_string_length(original);
    }
}
