//! Text normalization utilities.

mod domain;
mod language;
mod text;

pub use domain::{clean_address, clean_company, clean_name, normalize_email, normalize_url};
pub use language::{
    normalize_arabic, strip_arabic_diacritics, strip_bidi_marks, strip_hebrew_diacritics,
};
pub use text::{
    expand_abbreviations, fold_case, fold_case_locale, locale_aware_compare, normalize_unicode,
    normalize_whitespace, regex_replace, remove_stop_words, standardize_name, strip_diacritics,
    strip_punctuation, CaseFoldLocale, NormalizationForm,
};
