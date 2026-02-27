//! Built-in stop word lists and abbreviation tables.

use ahash::{AHashMap, AHashSet};

/// Returns the default English stop word set (127 common words).
#[must_use]
pub fn default_english_stop_words() -> AHashSet<String> {
    let words = [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "get",
        "got",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "its",
        "itself",
        "just",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "were",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ];
    words.iter().map(|w| (*w).to_string()).collect()
}

/// Returns the default address abbreviation table.
#[must_use]
pub fn default_address_abbreviations() -> AHashMap<String, String> {
    let pairs = [
        ("st.", "street"),
        ("ave.", "avenue"),
        ("blvd.", "boulevard"),
        ("rd.", "road"),
        ("dr.", "drive"),
        ("apt.", "apartment"),
        ("ln.", "lane"),
        ("ct.", "court"),
        ("pl.", "place"),
        ("pkwy.", "parkway"),
        ("hwy.", "highway"),
    ];
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// Returns the default company abbreviation table.
#[must_use]
pub fn default_company_abbreviations() -> AHashMap<String, String> {
    let pairs = [
        ("inc.", "incorporated"),
        ("inc", "incorporated"),
        ("corp.", "corporation"),
        ("corp", "corporation"),
        ("llc", "limited liability company"),
        ("ltd.", "limited"),
        ("ltd", "limited"),
        ("co.", "company"),
        ("co", "company"),
    ];
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// Returns the combined default abbreviation table (address + company).
#[must_use]
pub fn default_abbreviations() -> AHashMap<String, String> {
    let mut table = default_address_abbreviations();
    table.extend(default_company_abbreviations());
    table
}

/// Returns the set of name title prefixes (e.g. mr, mrs, dr).
#[must_use]
pub fn name_titles() -> AHashSet<String> {
    let words = [
        "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "prof", "prof.", "rev", "rev.",
        "sir", "dame", "hon", "hon.",
    ];
    words.iter().map(|w| (*w).to_string()).collect()
}

/// Returns the set of name suffixes (e.g. jr, sr, phd).
#[must_use]
pub fn name_suffixes() -> AHashSet<String> {
    let words = [
        "jr", "jr.", "sr", "sr.", "i", "ii", "iii", "iv", "v", "phd", "ph.d", "ph.d.", "md", "m.d",
        "m.d.", "esq", "esq.",
    ];
    words.iter().map(|w| (*w).to_string()).collect()
}

/// Returns the address normalization table (street types + directionals + unit types).
#[must_use]
pub fn address_normalization_table() -> AHashMap<String, String> {
    let pairs = [
        // Street types
        ("st", "street"),
        ("st.", "street"),
        ("ave", "avenue"),
        ("ave.", "avenue"),
        ("blvd", "boulevard"),
        ("blvd.", "boulevard"),
        ("rd", "road"),
        ("rd.", "road"),
        ("dr", "drive"),
        ("dr.", "drive"),
        ("ln", "lane"),
        ("ln.", "lane"),
        ("ct", "court"),
        ("ct.", "court"),
        ("pl", "place"),
        ("pl.", "place"),
        ("pkwy", "parkway"),
        ("pkwy.", "parkway"),
        ("hwy", "highway"),
        ("hwy.", "highway"),
        ("apt", "apartment"),
        ("apt.", "apartment"),
        ("cir", "circle"),
        ("cir.", "circle"),
        ("ter", "terrace"),
        ("ter.", "terrace"),
        ("trl", "trail"),
        ("trl.", "trail"),
        // Directionals
        ("n", "north"),
        ("n.", "north"),
        ("s", "south"),
        ("s.", "south"),
        ("e", "east"),
        ("e.", "east"),
        ("w", "west"),
        ("w.", "west"),
        ("ne", "northeast"),
        ("nw", "northwest"),
        ("se", "southeast"),
        ("sw", "southwest"),
        // Unit types
        ("ste", "suite"),
        ("ste.", "suite"),
        ("fl", "floor"),
        ("fl.", "floor"),
    ];
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// Returns the set of company legal suffixes (e.g. inc, corp, llc).
#[must_use]
pub fn company_legal_suffixes() -> AHashSet<String> {
    let words = [
        "inc",
        "inc.",
        "corp",
        "corp.",
        "llc",
        "ltd",
        "ltd.",
        "gmbh",
        "ag",
        "sa",
        "plc",
        "co",
        "co.",
        "lp",
        "llp",
        "corporation",
        "incorporated",
        "limited",
    ];
    words.iter().map(|w| (*w).to_string()).collect()
}

/// Returns the symbol → word replacement table for company names.
#[must_use]
pub fn company_symbol_replacements() -> AHashMap<String, String> {
    let pairs = [("&", "and"), ("+", "and")];
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_words_contains_common() {
        let sw = default_english_stop_words();
        assert!(sw.contains("the"));
        assert!(sw.contains("a"));
        assert!(sw.contains("is"));
        assert!(!sw.contains("hello"));
    }

    #[test]
    fn abbreviations_contains_entries() {
        let abbr = default_abbreviations();
        assert_eq!(abbr.get("inc.").unwrap(), "incorporated");
        assert_eq!(abbr.get("st.").unwrap(), "street");
        assert_eq!(abbr.get("llc").unwrap(), "limited liability company");
    }

    #[test]
    fn name_titles_contains_common() {
        let titles = name_titles();
        assert!(titles.contains("mr"));
        assert!(titles.contains("mr."));
        assert!(titles.contains("dr"));
        assert!(titles.contains("prof"));
        assert!(!titles.contains("john"));
    }

    #[test]
    fn name_suffixes_contains_common() {
        let suffixes = name_suffixes();
        assert!(suffixes.contains("jr"));
        assert!(suffixes.contains("jr."));
        assert!(suffixes.contains("phd"));
        assert!(suffixes.contains("iii"));
        assert!(!suffixes.contains("john"));
    }

    #[test]
    fn address_normalization_table_entries() {
        let table = address_normalization_table();
        assert_eq!(table.get("st").unwrap(), "street");
        assert_eq!(table.get("ave").unwrap(), "avenue");
        assert_eq!(table.get("nw").unwrap(), "northwest");
        assert_eq!(table.get("ste").unwrap(), "suite");
    }

    #[test]
    fn company_legal_suffixes_entries() {
        let suffixes = company_legal_suffixes();
        assert!(suffixes.contains("inc"));
        assert!(suffixes.contains("inc."));
        assert!(suffixes.contains("llc"));
        assert!(suffixes.contains("gmbh"));
        assert!(!suffixes.contains("acme"));
    }

    #[test]
    fn company_symbol_replacements_entries() {
        let replacements = company_symbol_replacements();
        assert_eq!(replacements.get("&").unwrap(), "and");
        assert_eq!(replacements.get("+").unwrap(), "and");
    }
}
