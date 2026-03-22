//! Native Polars expression plugin for reclink.
//!
//! Provides `#[polars_expr]` functions that operate directly on Arrow arrays,
//! avoiding Python GIL overhead. Requires the `polars-plugin` feature flag.
//!
//! # Expressions
//!
//! - `reclink_similarity` — compute string similarity between two columns
//! - `reclink_phonetic` — phonetic encoding of a column
//! - `reclink_match_best` — find best match from a list of candidates

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use reclink_core::metrics::{self, SimilarityMetric};
use reclink_core::phonetic::{self as phonetic_mod, PhoneticEncoder};

/// Kwargs for similarity expressions.
#[derive(Deserialize)]
struct SimilarityKwargs {
    scorer: String,
}

/// Kwargs for phonetic expressions.
#[derive(Deserialize)]
struct PhoneticKwargs {
    algorithm: String,
}

/// Kwargs for match_best expressions.
#[derive(Deserialize)]
struct MatchBestKwargs {
    candidates: Vec<String>,
    scorer: String,
    threshold: f64,
}

/// Compute string similarity between two string columns.
#[polars_expr(output_type=Float64)]
fn reclink_similarity(inputs: &[Series], kwargs: SimilarityKwargs) -> PolarsResult<Series> {
    let a = inputs[0].str()?;
    let b = inputs[1].str()?;

    let metric = metrics::metric_from_name(&kwargs.scorer)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    let out: Float64Chunked = a
        .into_iter()
        .zip(b.into_iter())
        .map(|(opt_a, opt_b)| match (opt_a, opt_b) {
            (Some(s_a), Some(s_b)) => Some(metric.similarity(s_a, s_b)),
            _ => None,
        })
        .collect();

    Ok(out.into_series())
}

/// Phonetic encoding of a string column.
#[polars_expr(output_type=String)]
fn reclink_phonetic(inputs: &[Series], kwargs: PhoneticKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;

    let encoder: Box<dyn PhoneticEncoder> = match kwargs.algorithm.as_str() {
        "soundex" => Box::new(phonetic_mod::Soundex),
        "metaphone" => Box::new(phonetic_mod::Metaphone),
        "double_metaphone" => Box::new(phonetic_mod::DoubleMetaphone),
        "nysiis" => Box::new(phonetic_mod::Nysiis),
        "caverphone" => Box::new(phonetic_mod::Caverphone),
        "cologne_phonetic" => Box::new(phonetic_mod::ColognePhonetic),
        "beider_morse" => Box::new(phonetic_mod::BeiderMorse::new()),
        other => {
            return Err(PolarsError::ComputeError(
                format!("unknown phonetic algorithm: {other}").into(),
            ));
        }
    };

    let out: StringChunked = ca
        .into_iter()
        .map(|opt_s| opt_s.map(|s| encoder.encode(s)))
        .collect();

    Ok(out.into_series())
}

/// Find best match from a list of candidates for each value in a string column.
///
/// Returns the best matching string (or null if below threshold).
#[polars_expr(output_type=String)]
fn reclink_match_best(inputs: &[Series], kwargs: MatchBestKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let candidates = &kwargs.candidates;
    let threshold = kwargs.threshold;

    let metric = metrics::metric_from_name(&kwargs.scorer)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    let out: StringChunked = ca
        .into_iter()
        .map(|opt_s| {
            opt_s.and_then(|s| {
                let mut best_score = threshold;
                let mut best_match: Option<&str> = None;

                for candidate in candidates {
                    let score = metric.similarity(s, candidate);
                    if score > best_score {
                        best_score = score;
                        best_match = Some(candidate.as_str());
                    }
                }

                best_match.map(String::from)
            })
        })
        .collect();

    Ok(out.into_series())
}
