//! Text preprocessing and normalization utilities.
//!
//! Provides Unicode normalization, case folding, whitespace normalization,
//! punctuation stripping, tokenization, and name standardization.

mod normalize;
pub mod stop_words;
mod tokenize;

pub use normalize::{
    clean_address, clean_company, clean_name, expand_abbreviations, fold_case, normalize_email,
    normalize_unicode, normalize_url, normalize_whitespace, regex_replace, remove_stop_words,
    standardize_name, strip_diacritics, strip_punctuation, NormalizationForm,
};
pub use tokenize::{ngram_tokenize, whitespace_tokenize};

use rayon::prelude::*;

use crate::error::Result;

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
    /// Remove common English stop words.
    RemoveStopWords,
    /// Expand common abbreviations (address + company).
    ExpandAbbreviations,
    /// Strip accents and diacritics.
    StripDiacritics,
    /// Apply a regex substitution.
    RegexReplace {
        pattern: String,
        replacement: String,
    },
    /// Clean a person name (remove titles, suffixes, comma-reorder).
    CleanName,
    /// Clean an address (expand street types, directionals, units).
    CleanAddress,
    /// Clean a company name (remove legal suffixes, replace symbols).
    CleanCompany,
    /// Normalize an email address (Gmail dot/plus handling, lowercase).
    NormalizeEmail,
    /// Normalize a URL (remove default ports, www, sort query params).
    NormalizeUrl,
    /// Expand synonyms using a user-supplied lookup table.
    SynonymExpand {
        table: ahash::AHashMap<String, String>,
    },
}

/// Applies a sequence of preprocessing operations to a string.
///
/// # Errors
///
/// Returns an error if a regex operation has an invalid pattern.
pub fn apply_ops(s: &str, ops: &[PreprocessOp]) -> Result<String> {
    let mut result = s.to_string();
    for op in ops {
        result = match op {
            PreprocessOp::FoldCase => fold_case(&result),
            PreprocessOp::NormalizeWhitespace => normalize_whitespace(&result),
            PreprocessOp::StripPunctuation => strip_punctuation(&result),
            PreprocessOp::StandardizeName => standardize_name(&result),
            PreprocessOp::NormalizeUnicode(form) => normalize_unicode(&result, *form),
            PreprocessOp::RemoveStopWords => {
                remove_stop_words(&result, &stop_words::default_english_stop_words())
            }
            PreprocessOp::ExpandAbbreviations => {
                expand_abbreviations(&result, &stop_words::default_abbreviations())
            }
            PreprocessOp::StripDiacritics => strip_diacritics(&result),
            PreprocessOp::RegexReplace {
                pattern,
                replacement,
            } => regex_replace(&result, pattern, replacement)?,
            PreprocessOp::CleanName => clean_name(&result),
            PreprocessOp::CleanAddress => clean_address(&result),
            PreprocessOp::CleanCompany => clean_company(&result),
            PreprocessOp::NormalizeEmail => normalize_email(&result),
            PreprocessOp::NormalizeUrl => normalize_url(&result),
            PreprocessOp::SynonymExpand { table } => expand_abbreviations(&result, table),
        };
    }
    Ok(result)
}

/// Applies preprocessing operations to a batch of strings in parallel.
///
/// # Errors
///
/// Returns an error if any regex operation has an invalid pattern.
pub fn preprocess_batch(strings: &[String], ops: &[PreprocessOp]) -> Result<Vec<String>> {
    strings.par_iter().map(|s| apply_ops(s, ops)).collect()
}
