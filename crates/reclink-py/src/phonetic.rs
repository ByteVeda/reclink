use pyo3::prelude::*;
use reclink_core::phonetic::{self as phonetic_mod, PhoneticEncoder};

/// Compute the Soundex code for a string.
#[pyfunction]
fn soundex(s: &str) -> String {
    phonetic_mod::Soundex.encode(s)
}

/// Compute the Metaphone code for a string.
#[pyfunction]
fn metaphone(s: &str) -> String {
    phonetic_mod::Metaphone.encode(s)
}

/// Compute the Double Metaphone codes for a string (primary, alternate).
#[pyfunction]
fn double_metaphone(s: &str) -> (String, String) {
    phonetic_mod::DoubleMetaphone.encode_both(s)
}

/// Compute the NYSIIS code for a string.
#[pyfunction]
fn nysiis(s: &str) -> String {
    phonetic_mod::Nysiis.encode(s)
}

/// Compute the Caverphone 2 code for a string.
#[pyfunction]
fn caverphone(s: &str) -> String {
    phonetic_mod::Caverphone.encode(s)
}

/// Compute the Cologne Phonetic code for a string.
#[pyfunction]
fn cologne_phonetic(s: &str) -> String {
    phonetic_mod::ColognePhonetic.encode(s)
}

/// Compute the Beider-Morse phonetic code for a string.
#[pyfunction]
#[pyo3(signature = (s, ashkenazi=false))]
fn beider_morse(s: &str, ashkenazi: bool) -> String {
    let encoder = if ashkenazi {
        phonetic_mod::BeiderMorse::ashkenazi()
    } else {
        phonetic_mod::BeiderMorse::new()
    };
    encoder.encode(s)
}

/// Compute the Phonex (improved Soundex) code for a string.
#[pyfunction]
fn phonex(s: &str) -> String {
    phonetic_mod::Phonex.encode(s)
}

/// Compute the Match Rating Approach code for a string.
#[pyfunction]
fn mra(s: &str) -> String {
    phonetic_mod::MatchRatingApproach.encode(s)
}

/// Compare two strings using the Match Rating Approach.
///
/// Returns True if the strings are considered a phonetic match.
#[pyfunction]
fn mra_compare(a: &str, b: &str) -> bool {
    phonetic_mod::mra_compare(a, b)
}

/// Compute the Daitch-Mokotoff Soundex code for a string.
///
/// Returns comma-separated codes when multiple alternatives exist.
#[pyfunction]
fn daitch_mokotoff(s: &str) -> String {
    phonetic_mod::DaitchMokotoff.encode(s)
}

/// Detect the most likely language/origin of a name.
///
/// Returns a string such as "english", "german", "french", "spanish",
/// "italian", "portuguese", "polish", "russian", "ashkenazi", or "generic".
#[pyfunction]
fn detect_language(s: &str) -> String {
    phonetic_mod::detect_language(s).to_string()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soundex, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(double_metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(nysiis, m)?)?;
    m.add_function(wrap_pyfunction!(caverphone, m)?)?;
    m.add_function(wrap_pyfunction!(cologne_phonetic, m)?)?;
    m.add_function(wrap_pyfunction!(beider_morse, m)?)?;
    m.add_function(wrap_pyfunction!(detect_language, m)?)?;
    m.add_function(wrap_pyfunction!(phonex, m)?)?;
    m.add_function(wrap_pyfunction!(mra, m)?)?;
    m.add_function(wrap_pyfunction!(mra_compare, m)?)?;
    m.add_function(wrap_pyfunction!(daitch_mokotoff, m)?)?;
    Ok(())
}
