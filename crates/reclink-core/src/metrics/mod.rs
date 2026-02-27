//! String similarity and distance metrics.
//!
//! This module provides 17 string comparison algorithms exposed through two traits:
//! - [`DistanceMetric`] for edit-distance style metrics (returning `usize`)
//! - [`SimilarityMetric`] for normalized similarity scores (returning `f64` in \[0, 1\])
//!
//! All algorithms are implemented from scratch with buffer reuse for batch operations.

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

use crate::error::{ReclinkError, Result};

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
    #[must_use]
    pub fn similarity(&self, a: &str, b: &str) -> f64 {
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
