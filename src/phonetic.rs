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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soundex, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(double_metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(nysiis, m)?)?;
    m.add_function(wrap_pyfunction!(caverphone, m)?)?;
    m.add_function(wrap_pyfunction!(cologne_phonetic, m)?)?;
    m.add_function(wrap_pyfunction!(beider_morse, m)?)?;
    Ok(())
}
