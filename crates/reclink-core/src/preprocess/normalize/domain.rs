//! Domain-specific cleaners for names, addresses, companies, emails, and URLs.

use super::text::expand_abbreviations;

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
