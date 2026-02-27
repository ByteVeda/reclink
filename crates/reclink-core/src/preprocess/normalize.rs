//! Text normalization utilities.

use ahash::{AHashMap, AHashSet};
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

/// Unicode normalization forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationForm {
    /// Canonical Decomposition, followed by Canonical Composition.
    Nfc,
    /// Canonical Decomposition.
    Nfd,
    /// Compatibility Decomposition, followed by Canonical Composition.
    Nfkc,
    /// Compatibility Decomposition.
    Nfkd,
}

/// Applies Unicode normalization to a string.
#[must_use]
pub fn normalize_unicode(s: &str, form: NormalizationForm) -> String {
    match form {
        NormalizationForm::Nfc => s.nfc().collect(),
        NormalizationForm::Nfd => s.nfd().collect(),
        NormalizationForm::Nfkc => s.nfkc().collect(),
        NormalizationForm::Nfkd => s.nfkd().collect(),
    }
}

/// Folds the string to lowercase.
#[must_use]
pub fn fold_case(s: &str) -> String {
    s.to_lowercase()
}

/// Normalizes whitespace: trims and collapses internal runs to single spaces.
#[must_use]
pub fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Strips all punctuation characters (Unicode category P).
#[must_use]
pub fn strip_punctuation(s: &str) -> String {
    s.chars().filter(|c| !c.is_ascii_punctuation()).collect()
}

/// Standardizes common name abbreviations and prefixes.
///
/// Expands abbreviations like "St." → "Saint", "Dr." → "Doctor", etc.
#[must_use]
pub fn standardize_name(s: &str) -> String {
    let replacements = [
        ("st.", "saint"),
        ("st ", "saint "),
        ("dr.", "doctor"),
        ("dr ", "doctor "),
        ("mr.", "mister"),
        ("mr ", "mister "),
        ("mrs.", "missus"),
        ("mrs ", "missus "),
        ("jr.", "junior"),
        ("jr ", "junior "),
        ("sr.", "senior"),
        ("sr ", "senior "),
    ];

    let mut result = s.to_lowercase();
    for (abbr, full) in &replacements {
        result = result.replace(abbr, full);
    }
    result
}

/// Removes stop words from a string.
///
/// Words are split on whitespace, filtered against the stop word set (case-insensitive),
/// and rejoined with single spaces.
#[must_use]
pub fn remove_stop_words(s: &str, stop_words: &AHashSet<String>) -> String {
    s.split_whitespace()
        .filter(|w| !stop_words.contains(&w.to_lowercase()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Strips accents/diacritics by decomposing to NFD and removing combining marks.
///
/// Uses Unicode canonical decomposition (NFD) to separate base characters from
/// their combining marks (category Mn), then filters out the marks.
#[must_use]
pub fn strip_diacritics(s: &str) -> String {
    s.nfd().filter(|c| !is_combining_mark(*c)).collect()
}

/// Applies a regex substitution to a string.
///
/// # Errors
///
/// Returns an error if the regex pattern is invalid.
pub fn regex_replace(s: &str, pattern: &str, replacement: &str) -> crate::error::Result<String> {
    let re = regex::Regex::new(pattern)
        .map_err(|e| crate::error::ReclinkError::InvalidConfig(format!("invalid regex: {e}")))?;
    Ok(re.replace_all(s, replacement).into_owned())
}

/// Expands abbreviations in a string using a lookup table.
///
/// Each whitespace-delimited token is checked (case-insensitive) against the table.
/// Matching tokens are replaced with their expansion.
#[must_use]
pub fn expand_abbreviations(s: &str, table: &AHashMap<String, String>) -> String {
    s.split_whitespace()
        .map(|w| {
            let lower = w.to_lowercase();
            table.get(&lower).cloned().unwrap_or_else(|| w.to_string())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Cleans a person name for matching.
///
/// 1. Lowercase
/// 2. Comma-reorder ("Last, First" → "First Last")
/// 3. Remove title prefixes (mr, dr, prof, etc.)
/// 4. Remove suffixes (jr, sr, iii, phd, etc.)
/// 5. Preserve hyphens, normalize whitespace
#[must_use]
pub fn clean_name(s: &str) -> String {
    let s = s.to_lowercase();
    // Comma-reorder: "Last, First Middle" → "First Middle Last"
    let s = if let Some((last, first)) = s.split_once(',') {
        format!("{} {}", first.trim(), last.trim())
    } else {
        s
    };
    let titles = crate::preprocess::stop_words::name_titles();
    let suffixes = crate::preprocess::stop_words::name_suffixes();
    let tokens: Vec<&str> = s
        .split_whitespace()
        .filter(|w| {
            let lower = w.to_lowercase();
            !titles.contains(&lower) && !suffixes.contains(&lower)
        })
        .collect();
    tokens.join(" ")
}

/// Cleans an address for matching.
///
/// 1. Lowercase
/// 2. Expand street types (st→street, ave→avenue, etc.)
/// 3. Expand directionals (n→north, nw→northwest, etc.)
/// 4. Normalize units (ste→suite)
#[must_use]
pub fn clean_address(s: &str) -> String {
    let s = s.to_lowercase();
    let table = crate::preprocess::stop_words::address_normalization_table();
    expand_abbreviations(&s, &table)
}

/// Cleans a company name for matching.
///
/// 1. Lowercase
/// 2. Replace symbols (&→and, +→and)
/// 3. Remove legal suffixes (inc, corp, llc, ltd, gmbh, etc.)
#[must_use]
pub fn clean_company(s: &str) -> String {
    let s = s.to_lowercase();
    // Replace symbols character-by-character
    let symbol_table = crate::preprocess::stop_words::company_symbol_replacements();
    let s: String = s
        .chars()
        .map(|c| {
            let cs = c.to_string();
            if let Some(replacement) = symbol_table.get(&cs) {
                replacement.clone()
            } else {
                cs
            }
        })
        .collect();
    let suffixes = crate::preprocess::stop_words::company_legal_suffixes();
    let tokens: Vec<&str> = s
        .split_whitespace()
        .filter(|w| !suffixes.contains(&w.to_lowercase()))
        .collect();
    tokens.join(" ")
}

/// Normalizes an email address for matching.
///
/// 1. Lowercase entire address
/// 2. Gmail/googlemail: remove plus-addressing, remove dots from local part,
///    normalize googlemail.com→gmail.com
/// 3. Non-Gmail: just lowercase
/// 4. Not an email (no @): return as-is
#[must_use]
pub fn normalize_email(s: &str) -> String {
    let s = s.to_lowercase();
    let Some((local, domain)) = s.split_once('@') else {
        return s;
    };
    let domain = if domain == "googlemail.com" {
        "gmail.com"
    } else {
        domain
    };
    if domain == "gmail.com" {
        // Remove plus-addressing
        let local = if let Some((base, _)) = local.split_once('+') {
            base
        } else {
            local
        };
        // Remove dots from local part
        let local: String = local.chars().filter(|c| *c != '.').collect();
        format!("{local}@{domain}")
    } else {
        format!("{local}@{domain}")
    }
}

/// Normalizes a URL for matching.
///
/// 1. Lowercase scheme + host
/// 2. Remove default ports (:80 http, :443 https)
/// 3. Remove www. prefix
/// 4. Remove trailing slashes from path
/// 5. Sort query parameters alphabetically
/// 6. Remove fragment (#)
/// 7. Manual string parsing (no `url` crate)
#[must_use]
pub fn normalize_url(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }

    // Split scheme
    let (scheme, rest) = if let Some(idx) = s.find("://") {
        let scheme = s[..idx].to_lowercase();
        (scheme, &s[idx + 3..])
    } else {
        return s.to_lowercase();
    };

    // Remove fragment
    let rest = if let Some(idx) = rest.find('#') {
        &rest[..idx]
    } else {
        rest
    };

    // Split host+port from path+query
    let (authority, path_and_query) = if let Some(idx) = rest.find('/') {
        (&rest[..idx], &rest[idx..])
    } else if let Some(idx) = rest.find('?') {
        (&rest[..idx], &rest[idx..])
    } else {
        (rest, "")
    };

    // Lowercase host, handle port
    let host = authority.to_lowercase();
    let host = match (scheme.as_str(), host.strip_suffix(":80")) {
        ("http", Some(h)) => h.to_string(),
        _ => match (scheme.as_str(), host.strip_suffix(":443")) {
            ("https", Some(h)) => h.to_string(),
            _ => host,
        },
    };

    // Remove www. prefix
    let host = host.strip_prefix("www.").unwrap_or(&host).to_string();

    // Split path and query
    let (path, query) = if let Some(idx) = path_and_query.find('?') {
        (&path_and_query[..idx], Some(&path_and_query[idx + 1..]))
    } else {
        (path_and_query, None)
    };

    // Remove trailing slashes from path
    let path = path.trim_end_matches('/');
    let path = if path.is_empty() { "" } else { path };

    // Sort query parameters
    let query_str = if let Some(q) = query {
        if q.is_empty() {
            String::new()
        } else {
            let mut params: Vec<&str> = q.split('&').collect();
            params.sort();
            format!("?{}", params.join("&"))
        }
    } else {
        String::new()
    };

    format!("{scheme}://{host}{path}{query_str}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_normalization() {
        let nfc = normalize_unicode("café", NormalizationForm::Nfc);
        let nfkc = normalize_unicode("café", NormalizationForm::Nfkc);
        assert_eq!(nfc, nfkc);
    }

    #[test]
    fn case_folding() {
        assert_eq!(fold_case("Hello WORLD"), "hello world");
    }

    #[test]
    fn whitespace_normalization() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
    }

    #[test]
    fn punctuation_stripping() {
        assert_eq!(strip_punctuation("hello, world!"), "hello world");
    }

    #[test]
    fn name_standardization() {
        assert_eq!(standardize_name("St. Louis"), "saint louis");
        assert_eq!(standardize_name("Dr. Smith"), "doctor smith");
    }

    #[test]
    fn stop_word_removal() {
        let sw = crate::preprocess::stop_words::default_english_stop_words();
        assert_eq!(remove_stop_words("the cat and the dog", &sw), "cat dog");
        assert_eq!(remove_stop_words("hello world", &sw), "hello world");
        assert_eq!(remove_stop_words("a", &sw), "");
    }

    #[test]
    fn abbreviation_expansion() {
        let table = crate::preprocess::stop_words::default_abbreviations();
        assert_eq!(
            expand_abbreviations("123 Main St.", &table),
            "123 Main street"
        );
        assert_eq!(
            expand_abbreviations("Acme Inc.", &table),
            "Acme incorporated"
        );
    }

    #[test]
    fn diacritic_stripping() {
        assert_eq!(strip_diacritics("café"), "cafe");
        assert_eq!(strip_diacritics("naïve"), "naive");
        assert_eq!(strip_diacritics("über"), "uber");
        assert_eq!(strip_diacritics("résumé"), "resume");
        // ASCII passthrough
        assert_eq!(strip_diacritics("hello"), "hello");
        assert_eq!(strip_diacritics(""), "");
    }

    #[test]
    fn regex_replacement() {
        assert_eq!(
            regex_replace("hello 123 world", r"\d+", "").unwrap(),
            "hello  world"
        );
        assert_eq!(
            regex_replace("foo-bar-baz", r"-", " ").unwrap(),
            "foo bar baz"
        );
        // Invalid regex
        assert!(regex_replace("test", r"[invalid", "").is_err());
    }

    #[test]
    fn clean_name_basic() {
        assert_eq!(clean_name("Mr. John Smith Jr."), "john smith");
        assert_eq!(clean_name("Dr. Jane Doe PhD"), "jane doe");
    }

    #[test]
    fn clean_name_comma_reorder() {
        assert_eq!(clean_name("Smith, John"), "john smith");
        assert_eq!(clean_name("Doe, Jane Marie"), "jane marie doe");
    }

    #[test]
    fn clean_name_preserves_hyphens() {
        assert_eq!(clean_name("Mary-Jane Watson"), "mary-jane watson");
    }

    #[test]
    fn clean_name_suffixes() {
        assert_eq!(clean_name("John Smith III"), "john smith");
        assert_eq!(clean_name("Robert Jones Sr."), "robert jones");
    }

    #[test]
    fn clean_address_basic() {
        assert_eq!(clean_address("123 Main St."), "123 main street");
        assert_eq!(clean_address("456 Oak Ave"), "456 oak avenue");
    }

    #[test]
    fn clean_address_directionals() {
        assert_eq!(clean_address("100 N Main St"), "100 north main street");
        assert_eq!(
            clean_address("200 NW Elm Blvd"),
            "200 northwest elm boulevard"
        );
    }

    #[test]
    fn clean_address_units() {
        assert_eq!(
            clean_address("300 First St Ste 100"),
            "300 first street suite 100"
        );
    }

    #[test]
    fn clean_company_basic() {
        assert_eq!(clean_company("Acme Inc."), "acme");
        assert_eq!(clean_company("Globex Corporation"), "globex");
    }

    #[test]
    fn clean_company_symbols() {
        assert_eq!(clean_company("Ben & Jerry's"), "ben and jerry's");
        assert_eq!(clean_company("A + B Corp"), "a and b");
    }

    #[test]
    fn clean_company_legal_suffixes() {
        assert_eq!(clean_company("Foo LLC"), "foo");
        assert_eq!(clean_company("Bar GmbH"), "bar");
        assert_eq!(clean_company("Baz Ltd."), "baz");
    }

    #[test]
    fn normalize_email_gmail() {
        assert_eq!(
            normalize_email("User.Name+tag@Gmail.COM"),
            "username@gmail.com"
        );
        assert_eq!(normalize_email("test@googlemail.com"), "test@gmail.com");
    }

    #[test]
    fn normalize_email_non_gmail() {
        assert_eq!(normalize_email("User@Example.COM"), "user@example.com");
    }

    #[test]
    fn normalize_email_no_at() {
        assert_eq!(normalize_email("not-an-email"), "not-an-email");
    }

    #[test]
    fn normalize_url_basic() {
        assert_eq!(
            normalize_url("HTTP://WWW.Example.COM/path/"),
            "http://example.com/path"
        );
    }

    #[test]
    fn normalize_url_default_ports() {
        assert_eq!(
            normalize_url("http://example.com:80/path"),
            "http://example.com/path"
        );
        assert_eq!(
            normalize_url("https://example.com:443/path"),
            "https://example.com/path"
        );
        // Non-default port preserved
        assert_eq!(
            normalize_url("http://example.com:8080/path"),
            "http://example.com:8080/path"
        );
    }

    #[test]
    fn normalize_url_sort_query() {
        assert_eq!(
            normalize_url("http://example.com/path?z=1&a=2"),
            "http://example.com/path?a=2&z=1"
        );
    }

    #[test]
    fn normalize_url_remove_fragment() {
        assert_eq!(
            normalize_url("http://example.com/path#section"),
            "http://example.com/path"
        );
    }

    #[test]
    fn normalize_url_empty() {
        assert_eq!(normalize_url(""), "");
    }
}
