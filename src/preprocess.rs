use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use reclink_core::preprocess;

/// Fold a string to lowercase.
#[pyfunction]
fn fold_case(s: &str) -> String {
    preprocess::fold_case(s)
}

/// Normalize whitespace (trim + collapse).
#[pyfunction]
fn normalize_whitespace(s: &str) -> String {
    preprocess::normalize_whitespace(s)
}

/// Strip ASCII punctuation from a string.
#[pyfunction]
fn strip_punctuation(s: &str) -> String {
    preprocess::strip_punctuation(s)
}

/// Standardize common name abbreviations.
#[pyfunction]
fn standardize_name(s: &str) -> String {
    preprocess::standardize_name(s)
}

/// Remove common English stop words from a string.
#[pyfunction]
fn remove_stop_words(s: &str) -> String {
    preprocess::remove_stop_words(
        s,
        &reclink_core::preprocess::stop_words::default_english_stop_words(),
    )
}

/// Expand common abbreviations (address + company) in a string.
#[pyfunction]
fn expand_abbreviations(s: &str) -> String {
    preprocess::expand_abbreviations(
        s,
        &reclink_core::preprocess::stop_words::default_abbreviations(),
    )
}

/// Strip accents and diacritics from a string.
///
/// Uses NFD decomposition to separate base characters from combining marks,
/// then removes the marks.
#[pyfunction]
fn strip_diacritics(s: &str) -> String {
    preprocess::strip_diacritics(s)
}

/// Apply a regex substitution to a string.
///
/// Replaces all matches of `pattern` with `replacement`.
#[pyfunction]
fn regex_replace(s: &str, pattern: &str, replacement: &str) -> PyResult<String> {
    preprocess::regex_replace(s, pattern, replacement)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Generate character n-grams from a string.
#[pyfunction]
#[pyo3(signature = (s, n=2))]
fn ngram_tokenize(s: &str, n: usize) -> Vec<String> {
    preprocess::ngram_tokenize(s, n)
}

/// Split a string on whitespace boundaries.
#[pyfunction]
fn whitespace_tokenize(s: &str) -> Vec<String> {
    preprocess::whitespace_tokenize(s)
        .into_iter()
        .map(String::from)
        .collect()
}

/// Apply Unicode normalization to a string.
#[pyfunction]
#[pyo3(signature = (s, form="nfkc"))]
fn normalize_unicode(s: &str, form: &str) -> PyResult<String> {
    let nf = match form {
        "nfc" => preprocess::NormalizationForm::Nfc,
        "nfd" => preprocess::NormalizationForm::Nfd,
        "nfkc" => preprocess::NormalizationForm::Nfkc,
        "nfkd" => preprocess::NormalizationForm::Nfkd,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown normalization form: {form}. Expected: nfc, nfd, nfkc, nfkd"
            )));
        }
    };
    Ok(preprocess::normalize_unicode(s, nf))
}

/// Clean a person name for matching.
///
/// Lowercases, reorders "Last, First" to "First Last", removes title prefixes
/// (mr, dr, prof, etc.) and suffixes (jr, sr, iii, phd, etc.), preserves hyphens.
#[pyfunction]
fn clean_name(s: &str) -> String {
    preprocess::clean_name(s)
}

/// Clean an address for matching.
///
/// Lowercases, expands street types (st→street, ave→avenue), directionals
/// (n→north, nw→northwest), and unit types (ste→suite).
#[pyfunction]
fn clean_address(s: &str) -> String {
    preprocess::clean_address(s)
}

/// Clean a company name for matching.
///
/// Lowercases, replaces symbols (&→and, +→and), removes legal suffixes
/// (inc, corp, llc, ltd, gmbh, etc.).
#[pyfunction]
fn clean_company(s: &str) -> String {
    preprocess::clean_company(s)
}

/// Normalize an email address for matching.
///
/// Lowercases the entire address. For Gmail/googlemail addresses: removes
/// plus-addressing, removes dots from local part, normalizes googlemail→gmail.
/// Returns non-email strings as-is.
#[pyfunction]
fn normalize_email(s: &str) -> String {
    preprocess::normalize_email(s)
}

/// Normalize a URL for matching.
///
/// Lowercases scheme + host, removes default ports (:80/:443), removes www. prefix,
/// removes trailing slashes, sorts query parameters, removes fragments.
#[pyfunction]
fn normalize_url(s: &str) -> String {
    preprocess::normalize_url(s)
}

/// Expand synonyms in a string using a lookup table.
///
/// Each whitespace-delimited token is checked (case-insensitive) against the table.
/// Matching tokens are replaced with their expansion.
#[pyfunction]
fn synonym_expand(s: &str, table: std::collections::HashMap<String, String>) -> String {
    let ahash_table: ahash::AHashMap<String, String> = table.into_iter().collect();
    preprocess::expand_abbreviations(s, &ahash_table)
}

/// Transliterate Cyrillic characters to Latin equivalents.
///
/// Non-Cyrillic characters (including Latin) are passed through unchanged.
#[pyfunction]
fn transliterate_cyrillic(s: &str) -> String {
    reclink_core::preprocess::transliterate::transliterate(
        s,
        reclink_core::preprocess::transliterate::Script::Cyrillic,
    )
}

/// Transliterate Greek characters to Latin equivalents.
///
/// Non-Greek characters (including Latin) are passed through unchanged.
#[pyfunction]
fn transliterate_greek(s: &str) -> String {
    reclink_core::preprocess::transliterate::transliterate(
        s,
        reclink_core::preprocess::transliterate::Script::Greek,
    )
}

/// Transliterate Arabic characters to Latin equivalents.
///
/// Non-Arabic characters (including Latin) are passed through unchanged.
#[pyfunction]
fn transliterate_arabic(s: &str) -> String {
    reclink_core::preprocess::transliterate::transliterate(
        s,
        reclink_core::preprocess::transliterate::Script::Arabic,
    )
}

/// Transliterate Hebrew characters to Latin equivalents.
///
/// Non-Hebrew characters (including Latin) are passed through unchanged.
#[pyfunction]
fn transliterate_hebrew(s: &str) -> String {
    reclink_core::preprocess::transliterate::transliterate(
        s,
        reclink_core::preprocess::transliterate::Script::Hebrew,
    )
}

/// Strip Arabic diacritics (harakat, tatweel, superscript alef).
#[pyfunction]
fn strip_arabic_diacritics(s: &str) -> String {
    preprocess::strip_arabic_diacritics(s)
}

/// Strip Hebrew diacritics (niqqud, cantillation marks).
#[pyfunction]
fn strip_hebrew_diacritics(s: &str) -> String {
    preprocess::strip_hebrew_diacritics(s)
}

/// Normalize Arabic text (alef variants → bare alef, teh marbuta → heh).
#[pyfunction]
fn normalize_arabic(s: &str) -> String {
    preprocess::normalize_arabic(s)
}

/// Strip Unicode bidirectional control marks.
#[pyfunction]
fn strip_bidi_marks(s: &str) -> String {
    preprocess::strip_bidi_marks(s)
}

/// Transliterate Devanagari characters to Latin equivalents (IAST-style).
///
/// Virama-aware: consonant clusters are handled correctly.
/// Non-Devanagari characters are passed through unchanged.
#[pyfunction]
fn transliterate_devanagari(s: &str) -> String {
    reclink_core::preprocess::transliterate::transliterate(
        s,
        reclink_core::preprocess::transliterate::Script::Devanagari,
    )
}

/// Transliterate Hangul characters to Latin equivalents (Revised Romanization).
///
/// Algorithmically decomposes syllable blocks into initial/medial/final components.
/// Non-Hangul characters are passed through unchanged.
#[pyfunction]
fn transliterate_hangul(s: &str) -> String {
    reclink_core::preprocess::transliterate::transliterate(
        s,
        reclink_core::preprocess::transliterate::Script::Hangul,
    )
}

/// Apply a chain of preprocessing operations to a batch of strings in parallel.
#[pyfunction]
fn preprocess_batch(
    py: Python<'_>,
    strings: Vec<String>,
    operations: Vec<String>,
) -> PyResult<Vec<String>> {
    let ops = crate::parsers::parse_preprocess_ops(&operations)?;
    py.allow_threads(|| {
        preprocess::preprocess_batch(&strings, &ops)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// Generate character n-grams for a batch of strings in parallel.
#[pyfunction]
#[pyo3(signature = (strings, n=2))]
fn ngram_tokenize_batch(py: Python<'_>, strings: Vec<String>, n: usize) -> Vec<Vec<String>> {
    py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| preprocess::ngram_tokenize(s, n))
            .collect()
    })
}

/// Split each string on whitespace for a batch of strings in parallel.
#[pyfunction]
fn whitespace_tokenize_batch(py: Python<'_>, strings: Vec<String>) -> Vec<Vec<String>> {
    py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| {
                preprocess::whitespace_tokenize(s)
                    .into_iter()
                    .map(String::from)
                    .collect()
            })
            .collect()
    })
}

/// Tokenize a string into individual characters (whitespace removed).
///
/// Useful for CJK text where whitespace delimiters are absent.
#[pyfunction]
fn character_tokenize(s: &str) -> Vec<String> {
    preprocess::character_tokenize(s)
}

/// Smart tokenizer that auto-detects CJK vs Latin runs.
///
/// CJK characters become individual tokens; Latin runs are whitespace-split.
#[pyfunction]
fn smart_tokenize(s: &str) -> Vec<String> {
    preprocess::smart_tokenize(s)
}

/// Tokenize each string into individual characters in parallel.
#[pyfunction]
fn character_tokenize_batch(py: Python<'_>, strings: Vec<String>) -> Vec<Vec<String>> {
    py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| preprocess::character_tokenize(s))
            .collect()
    })
}

/// Generate character n-grams from CJK text.
///
/// Extracts CJK characters and produces n-grams of the specified size.
#[pyfunction]
#[pyo3(signature = (s, n=2))]
fn cjk_ngram_tokenize(s: &str, n: usize) -> Vec<String> {
    preprocess::cjk_ngram_tokenize(s, n)
}

/// Smart n-gram tokenizer for mixed CJK/Latin text.
///
/// CJK characters become character n-grams; Latin runs become whitespace tokens.
#[pyfunction]
#[pyo3(signature = (s, n=2))]
fn smart_tokenize_ngram(s: &str, n: usize) -> Vec<String> {
    preprocess::smart_tokenize_ngram(s, n)
}

/// Generate CJK character n-grams for a batch of strings in parallel.
#[pyfunction]
#[pyo3(signature = (strings, n=2))]
fn cjk_ngram_tokenize_batch(py: Python<'_>, strings: Vec<String>, n: usize) -> Vec<Vec<String>> {
    py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| preprocess::cjk_ngram_tokenize(s, n))
            .collect()
    })
}

/// Smart n-gram tokenize a batch of strings in parallel.
#[pyfunction]
#[pyo3(signature = (strings, n=2))]
fn smart_tokenize_ngram_batch(py: Python<'_>, strings: Vec<String>, n: usize) -> Vec<Vec<String>> {
    py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| preprocess::smart_tokenize_ngram(s, n))
            .collect()
    })
}

/// Smart-tokenize each string in parallel.
#[pyfunction]
fn smart_tokenize_batch(py: Python<'_>, strings: Vec<String>) -> Vec<Vec<String>> {
    py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| preprocess::smart_tokenize(s))
            .collect()
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fold_case, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_whitespace, m)?)?;
    m.add_function(wrap_pyfunction!(strip_punctuation, m)?)?;
    m.add_function(wrap_pyfunction!(standardize_name, m)?)?;
    m.add_function(wrap_pyfunction!(remove_stop_words, m)?)?;
    m.add_function(wrap_pyfunction!(expand_abbreviations, m)?)?;
    m.add_function(wrap_pyfunction!(strip_diacritics, m)?)?;
    m.add_function(wrap_pyfunction!(regex_replace, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(whitespace_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_unicode, m)?)?;
    m.add_function(wrap_pyfunction!(clean_name, m)?)?;
    m.add_function(wrap_pyfunction!(clean_address, m)?)?;
    m.add_function(wrap_pyfunction!(clean_company, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_email, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_url, m)?)?;
    m.add_function(wrap_pyfunction!(synonym_expand, m)?)?;
    m.add_function(wrap_pyfunction!(transliterate_cyrillic, m)?)?;
    m.add_function(wrap_pyfunction!(transliterate_greek, m)?)?;
    m.add_function(wrap_pyfunction!(transliterate_arabic, m)?)?;
    m.add_function(wrap_pyfunction!(transliterate_hebrew, m)?)?;
    m.add_function(wrap_pyfunction!(strip_arabic_diacritics, m)?)?;
    m.add_function(wrap_pyfunction!(strip_hebrew_diacritics, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_arabic, m)?)?;
    m.add_function(wrap_pyfunction!(strip_bidi_marks, m)?)?;
    m.add_function(wrap_pyfunction!(transliterate_devanagari, m)?)?;
    m.add_function(wrap_pyfunction!(transliterate_hangul, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(whitespace_tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(character_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(smart_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(character_tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(smart_tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cjk_ngram_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(smart_tokenize_ngram, m)?)?;
    m.add_function(wrap_pyfunction!(cjk_ngram_tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(smart_tokenize_ngram_batch, m)?)?;
    Ok(())
}
