//! Preprocessing functions exposed to JavaScript.

use reclink_core::preprocess::*;
use wasm_bindgen::prelude::*;

fn op_from_name(name: &str) -> Result<PreprocessOp, JsError> {
    match name {
        "fold_case" => Ok(PreprocessOp::FoldCase),
        "normalize_whitespace" => Ok(PreprocessOp::NormalizeWhitespace),
        "strip_punctuation" => Ok(PreprocessOp::StripPunctuation),
        "standardize_name" => Ok(PreprocessOp::StandardizeName),
        "normalize_unicode_nfc" => Ok(PreprocessOp::NormalizeUnicode(NormalizationForm::Nfc)),
        "normalize_unicode_nfkc" => Ok(PreprocessOp::NormalizeUnicode(NormalizationForm::Nfkc)),
        "remove_stop_words" => Ok(PreprocessOp::RemoveStopWords),
        "expand_abbreviations" => Ok(PreprocessOp::ExpandAbbreviations),
        "strip_diacritics" => Ok(PreprocessOp::StripDiacritics),
        "clean_name" => Ok(PreprocessOp::CleanName),
        "clean_address" => Ok(PreprocessOp::CleanAddress),
        "clean_company" => Ok(PreprocessOp::CleanCompany),
        "normalize_email" => Ok(PreprocessOp::NormalizeEmail),
        "normalize_url" => Ok(PreprocessOp::NormalizeUrl),
        "strip_arabic_diacritics" => Ok(PreprocessOp::StripArabicDiacritics),
        "strip_hebrew_diacritics" => Ok(PreprocessOp::StripHebrewDiacritics),
        "normalize_arabic" => Ok(PreprocessOp::NormalizeArabic),
        "strip_bidi_marks" => Ok(PreprocessOp::StripBidiMarks),
        _ => Err(JsError::new(&format!("unknown preprocessing op: '{name}'"))),
    }
}

/// Apply a sequence of preprocessing operations to a string.
///
/// `ops` is a JS array of operation name strings.
#[wasm_bindgen]
pub fn preprocess(s: &str, ops: JsValue) -> Result<String, JsError> {
    let op_names: Vec<String> =
        serde_wasm_bindgen::from_value(ops).map_err(|e| JsError::new(&e.to_string()))?;
    let parsed: Vec<PreprocessOp> = op_names
        .iter()
        .map(|name| op_from_name(name))
        .collect::<Result<Vec<_>, _>>()?;
    apply_ops(s, &parsed).map_err(|e| JsError::new(&e.to_string()))
}

/// Apply a single preprocessing operation and return the result.
/// Useful for step-by-step pipeline visualization.
#[wasm_bindgen]
pub fn preprocess_step(s: &str, op: &str) -> Result<String, JsError> {
    let parsed = op_from_name(op)?;
    apply_ops(s, &[parsed]).map_err(|e| JsError::new(&e.to_string()))
}

/// List all available preprocessing operation names.
#[wasm_bindgen]
pub fn list_preprocess_ops() -> JsValue {
    let names = vec![
        "fold_case",
        "normalize_whitespace",
        "strip_punctuation",
        "standardize_name",
        "normalize_unicode_nfc",
        "normalize_unicode_nfkc",
        "remove_stop_words",
        "expand_abbreviations",
        "strip_diacritics",
        "clean_name",
        "clean_address",
        "clean_company",
        "normalize_email",
        "normalize_url",
        "strip_arabic_diacritics",
        "strip_hebrew_diacritics",
        "normalize_arabic",
        "strip_bidi_marks",
    ];
    serde_wasm_bindgen::to_value(&names).unwrap()
}
