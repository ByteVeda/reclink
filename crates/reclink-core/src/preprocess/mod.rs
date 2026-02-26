//! Text preprocessing and normalization utilities.
//!
//! Provides Unicode normalization, case folding, whitespace normalization,
//! punctuation stripping, tokenization, and name standardization.

mod normalize;
mod tokenize;

pub use normalize::{
    fold_case, normalize_unicode, normalize_whitespace, standardize_name, strip_punctuation,
    NormalizationForm,
};
pub use tokenize::{ngram_tokenize, whitespace_tokenize};

use rayon::prelude::*;

/// A composable preprocessing operation.
#[derive(Debug, Clone)]
pub enum PreprocessOp {
    /// Fold to lowercase.
    FoldCase,
    /// Trim + collapse whitespace.
    NormalizeWhitespace,
    /// Strip ASCII punctuation.
    StripPunctuation,
    /// Expand common name abbreviations.
    StandardizeName,
    /// Apply Unicode normalization.
    NormalizeUnicode(NormalizationForm),
}

/// Applies a sequence of preprocessing operations to a string.
#[must_use]
pub fn apply_ops(s: &str, ops: &[PreprocessOp]) -> String {
    let mut result = s.to_string();
    for op in ops {
        result = match op {
            PreprocessOp::FoldCase => fold_case(&result),
            PreprocessOp::NormalizeWhitespace => normalize_whitespace(&result),
            PreprocessOp::StripPunctuation => strip_punctuation(&result),
            PreprocessOp::StandardizeName => standardize_name(&result),
            PreprocessOp::NormalizeUnicode(form) => normalize_unicode(&result, *form),
        };
    }
    result
}

/// Applies preprocessing operations to a batch of strings in parallel.
#[must_use]
pub fn preprocess_batch(strings: &[String], ops: &[PreprocessOp]) -> Vec<String> {
    strings.par_iter().map(|s| apply_ops(s, ops)).collect()
}
