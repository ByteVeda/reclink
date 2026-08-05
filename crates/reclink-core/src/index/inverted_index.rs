//! Inverted index for token-based candidate retrieval.
//!
//! Maps tokens to record indices for fast lookup of records sharing
//! common tokens with a query string.

use ahash::AHashMap;

/// Tokenization strategy for the inverted index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TokenizerKind {
    /// Split on whitespace.
    Whitespace,
    /// Character n-grams of size n.
    Ngram(usize),
}

/// Result from an inverted index search.
#[derive(Debug, Clone)]
pub struct InvertedSearchResult {
    /// The matched string.
    pub value: String,
    /// Index in the original build list.
    pub index: usize,
    /// Number of tokens shared with the query.
    pub shared_tokens: usize,
}

/// Token-to-record-indices mapping for fast candidate retrieval.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InvertedIndex {
    index: AHashMap<String, Vec<usize>>,
    records: Vec<String>,
    tokenizer: TokenizerKind,
}

impl InvertedIndex {
    /// Builds an inverted index from a list of strings.
    #[must_use]
    pub fn build(strings: &[&str], tokenizer: TokenizerKind) -> Self {
        let mut index: AHashMap<String, Vec<usize>> = AHashMap::new();
        let records: Vec<String> = strings.iter().map(|s| (*s).to_string()).collect();

        for (i, s) in strings.iter().enumerate() {
            let tokens = tokenize(s, &tokenizer);
            for token in tokens {
                index.entry(token).or_default().push(i);
            }
        }

        Self {
            index,
            records,
            tokenizer,
        }
    }

    /// Searches for records sharing at least `min_shared` tokens with the query.
    #[must_use]
    pub fn search(&self, query: &str, min_shared: usize) -> Vec<InvertedSearchResult> {
        let tokens = tokenize(query, &self.tokenizer);
        let mut counts: AHashMap<usize, usize> = AHashMap::new();

        for token in &tokens {
            if let Some(indices) = self.index.get(token) {
                for &idx in indices {
                    *counts.entry(idx).or_insert(0) += 1;
                }
            }
        }

        let mut results: Vec<InvertedSearchResult> = counts
            .into_iter()
            .filter(|&(_, count)| count >= min_shared)
            .map(|(idx, count)| InvertedSearchResult {
                value: self.records[idx].clone(),
                index: idx,
                shared_tokens: count,
            })
            .collect();

        results.sort_by(|a, b| b.shared_tokens.cmp(&a.shared_tokens));
        results
    }

    /// Returns the top-k records by number of shared tokens.
    #[must_use]
    pub fn search_top_k(&self, query: &str, k: usize) -> Vec<InvertedSearchResult> {
        let mut results = self.search(query, 1);
        results.truncate(k);
        results
    }

    /// Returns the number of indexed records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns the number of unique tokens in the index.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.index.len()
    }
}

fn tokenize(s: &str, kind: &TokenizerKind) -> Vec<String> {
    match kind {
        TokenizerKind::Whitespace => s.split_whitespace().map(|t| t.to_lowercase()).collect(),
        TokenizerKind::Ngram(n) => {
            let chars: Vec<char> = s.chars().collect();
            if chars.len() < *n {
                return vec![];
            }
            chars
                .windows(*n)
                .map(|w| w.iter().collect::<String>())
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_basic() {
        let strings = vec!["hello world", "world peace", "hello peace"];
        let idx = InvertedIndex::build(&strings, TokenizerKind::Whitespace);
        let results = idx.search("hello world", 1);
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.index == 0 && r.shared_tokens == 2));
    }

    #[test]
    fn ngram_basic() {
        let strings = vec!["hello", "hallo", "world"];
        let idx = InvertedIndex::build(&strings, TokenizerKind::Ngram(2));
        let results = idx.search("hello", 1);
        assert!(results.iter().any(|r| r.index == 0));
    }

    #[test]
    fn min_shared_filter() {
        let strings = vec!["hello world", "hello", "world"];
        let idx = InvertedIndex::build(&strings, TokenizerKind::Whitespace);
        let results = idx.search("hello world", 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn top_k() {
        let strings = vec!["a b c", "a b", "a", "x y z"];
        let idx = InvertedIndex::build(&strings, TokenizerKind::Whitespace);
        let results = idx.search_top_k("a b c", 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn empty_index() {
        let idx = InvertedIndex::build(&[], TokenizerKind::Whitespace);
        assert!(idx.is_empty());
        assert!(idx.search("hello", 1).is_empty());
    }

    #[test]
    fn vocab_size() {
        let strings = vec!["hello world", "hello"];
        let idx = InvertedIndex::build(&strings, TokenizerKind::Whitespace);
        assert_eq!(idx.vocab_size(), 2); // "hello", "world"
    }
}
