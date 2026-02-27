use ahash::AHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use reclink_core::phonetic::{self as phonetic_mod};
use reclink_core::preprocess;

pub fn parse_preprocess_ops(names: &[String]) -> PyResult<Vec<preprocess::PreprocessOp>> {
    names
        .iter()
        .map(|name| match name.as_str() {
            "fold_case" => Ok(preprocess::PreprocessOp::FoldCase),
            "normalize_whitespace" => Ok(preprocess::PreprocessOp::NormalizeWhitespace),
            "strip_punctuation" => Ok(preprocess::PreprocessOp::StripPunctuation),
            "standardize_name" => Ok(preprocess::PreprocessOp::StandardizeName),
            "normalize_unicode_nfc" => Ok(preprocess::PreprocessOp::NormalizeUnicode(
                preprocess::NormalizationForm::Nfc,
            )),
            "normalize_unicode_nfd" => Ok(preprocess::PreprocessOp::NormalizeUnicode(
                preprocess::NormalizationForm::Nfd,
            )),
            "normalize_unicode_nfkc" => Ok(preprocess::PreprocessOp::NormalizeUnicode(
                preprocess::NormalizationForm::Nfkc,
            )),
            "normalize_unicode_nfkd" => Ok(preprocess::PreprocessOp::NormalizeUnicode(
                preprocess::NormalizationForm::Nfkd,
            )),
            "remove_stop_words" => Ok(preprocess::PreprocessOp::RemoveStopWords),
            "expand_abbreviations" => Ok(preprocess::PreprocessOp::ExpandAbbreviations),
            "strip_diacritics" => Ok(preprocess::PreprocessOp::StripDiacritics),
            "clean_name" => Ok(preprocess::PreprocessOp::CleanName),
            "clean_address" => Ok(preprocess::PreprocessOp::CleanAddress),
            "clean_company" => Ok(preprocess::PreprocessOp::CleanCompany),
            "normalize_email" => Ok(preprocess::PreprocessOp::NormalizeEmail),
            "normalize_url" => Ok(preprocess::PreprocessOp::NormalizeUrl),
            other if other.starts_with("synonym_expand:") => {
                let json = &other["synonym_expand:".len()..];
                let map: std::collections::HashMap<String, String> = serde_json::from_str(json)
                    .map_err(|e| {
                        PyValueError::new_err(format!("synonym_expand: invalid JSON table: {e}"))
                    })?;
                let table: AHashMap<String, String> = map.into_iter().collect();
                Ok(preprocess::PreprocessOp::SynonymExpand { table })
            }
            other if other.starts_with("regex_replace:") => {
                let parts: Vec<&str> = other.splitn(3, ':').collect();
                if parts.len() != 3 {
                    return Err(PyValueError::new_err(
                        "regex_replace format: 'regex_replace:<pattern>:<replacement>'",
                    ));
                }
                Ok(preprocess::PreprocessOp::RegexReplace {
                    pattern: parts[1].to_string(),
                    replacement: parts[2].to_string(),
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "unknown operation: {name}. Expected: fold_case, normalize_whitespace, \
                 strip_punctuation, standardize_name, normalize_unicode_nfc, \
                 normalize_unicode_nfd, normalize_unicode_nfkc, normalize_unicode_nfkd, \
                 remove_stop_words, expand_abbreviations, strip_diacritics, \
                 clean_name, clean_address, clean_company, \
                 normalize_email, normalize_url, \
                 synonym_expand:{{\"key\":\"value\"}}, \
                 regex_replace:<pattern>:<replacement>"
            ))),
        })
        .collect()
}

pub fn parse_phonetic_algorithm(name: &str) -> PyResult<phonetic_mod::PhoneticAlgorithm> {
    match name {
        "soundex" => Ok(phonetic_mod::PhoneticAlgorithm::Soundex(
            phonetic_mod::Soundex,
        )),
        "metaphone" => Ok(phonetic_mod::PhoneticAlgorithm::Metaphone(
            phonetic_mod::Metaphone,
        )),
        "double_metaphone" => Ok(phonetic_mod::PhoneticAlgorithm::DoubleMetaphone(
            phonetic_mod::DoubleMetaphone,
        )),
        "nysiis" => Ok(phonetic_mod::PhoneticAlgorithm::Nysiis(
            phonetic_mod::Nysiis,
        )),
        "caverphone" => Ok(phonetic_mod::PhoneticAlgorithm::Caverphone(
            phonetic_mod::Caverphone,
        )),
        "cologne_phonetic" => Ok(phonetic_mod::PhoneticAlgorithm::ColognePhonetic(
            phonetic_mod::ColognePhonetic,
        )),
        "beider_morse" => Ok(phonetic_mod::PhoneticAlgorithm::BeiderMorse(
            phonetic_mod::BeiderMorse::new(),
        )),
        _ => Err(PyValueError::new_err(format!(
            "unknown phonetic algorithm: {name}"
        ))),
    }
}

pub fn parse_date_resolution(name: &str) -> PyResult<reclink_core::blocking::DateResolution> {
    match name {
        "year" => Ok(reclink_core::blocking::DateResolution::Year),
        "month" => Ok(reclink_core::blocking::DateResolution::Month),
        "day" => Ok(reclink_core::blocking::DateResolution::Day),
        _ => Err(PyValueError::new_err(format!(
            "unknown date resolution: {name}. Expected: year, month, day"
        ))),
    }
}
