//! String similarity and distance metrics.
//!
//! This module provides 8 string comparison algorithms exposed through two traits:
//! - [`DistanceMetric`] for edit-distance style metrics (returning `usize`)
//! - [`SimilarityMetric`] for normalized similarity scores (returning `f64` in \[0, 1\])
//!
//! All algorithms are implemented from scratch with buffer reuse for batch operations.

pub mod cosine;
pub mod damerau_levenshtein;
pub mod hamming;
pub mod jaccard;
pub mod jaro;
pub mod jaro_winkler;
pub mod levenshtein;
pub mod sorensen_dice;

pub use cosine::Cosine;
pub use damerau_levenshtein::DamerauLevenshtein;
pub use hamming::Hamming;
pub use jaccard::Jaccard;
pub use jaro::Jaro;
pub use jaro_winkler::JaroWinkler;
pub use levenshtein::Levenshtein;
pub use sorensen_dice::SorensenDice;

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
#[derive(Debug, Clone)]
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
/// `"jaro"`, `"jaro_winkler"`, `"cosine"`, `"jaccard"`, `"sorensen_dice"`.
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
        _ => Err(ReclinkError::InvalidConfig(format!(
            "unknown metric: `{name}`. Expected one of: levenshtein, damerau_levenshtein, \
             hamming, jaro, jaro_winkler, cosine, jaccard, sorensen_dice"
        ))),
    }
}
