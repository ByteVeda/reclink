//! Inverted n-gram index for fast approximate string matching.
//!
//! Maps each character n-gram to the set of strings containing it.
//! At query time, counts how many n-grams the query shares with each candidate
//! and returns those above a threshold.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::preprocess::ngram_tokenize;

/// Result from an n-gram index search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramSearchResult {
    /// The matched string value.
    pub value: String,
    /// Index of the string in the original build list.
    pub index: usize,
    /// Number of shared n-grams with the query.
    pub shared_ngrams: usize,
}

/// An inverted n-gram index for fast approximate string matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramIndex {
    index: AHashMap<String, Vec<usize>>,
    strings: Vec<String>,
    n: usize,
}

impl NgramIndex {
    /// Build an n-gram index from a list of strings.
    #[must_use]
    pub fn build(strings: &[&str], n: usize) -> Self {
        let mut index: AHashMap<String, Vec<usize>> = AHashMap::new();
        let owned: Vec<String> = strings.iter().map(|s| s.to_string()).collect();

        for (i, s) in strings.iter().enumerate() {
            let ngrams = ngram_tokenize(s, n);
            for ng in ngrams {
                index.entry(ng).or_default().push(i);
            }
        }

        Self {
            index,
            strings: owned,
            n,
        }
    }

    /// Find all strings sharing at least `threshold` n-grams with the query.
    #[must_use]
    pub fn search(&self, query: &str, threshold: usize) -> Vec<NgramSearchResult> {
        let query_ngrams = ngram_tokenize(query, self.n);
        let counts = self.count_shared(&query_ngrams);

        let mut results: Vec<NgramSearchResult> = counts
            .into_iter()
            .filter(|&(_, count)| count >= threshold)
            .map(|(idx, count)| NgramSearchResult {
                value: self.strings[idx].clone(),
                index: idx,
                shared_ngrams: count,
            })
            .collect();

        results.sort_by(|a, b| b.shared_ngrams.cmp(&a.shared_ngrams));
        results
    }

    /// Find the k strings sharing the most n-grams with the query.
    #[must_use]
    pub fn search_top_k(&self, query: &str, k: usize) -> Vec<NgramSearchResult> {
        let query_ngrams = ngram_tokenize(query, self.n);
        let counts = self.count_shared(&query_ngrams);

        let mut results: Vec<NgramSearchResult> = counts
            .into_iter()
            .map(|(idx, count)| NgramSearchResult {
                value: self.strings[idx].clone(),
                index: idx,
                shared_ngrams: count,
            })
            .collect();

        results.sort_by(|a, b| b.shared_ngrams.cmp(&a.shared_ngrams));
        results.truncate(k);
        results
    }

    /// Returns the number of strings in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    fn count_shared(&self, query_ngrams: &[String]) -> Vec<(usize, usize)> {
        let mut counts: AHashMap<usize, usize> = AHashMap::new();
        for ng in query_ngrams {
            if let Some(indices) = self.index.get(ng) {
                for &idx in indices {
                    *counts.entry(idx).or_insert(0) += 1;
                }
            }
        }
        counts.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_search() {
        let index = NgramIndex::build(&["hello", "help", "world"], 2);
        let results = index.search("hello", 2);
        // "hello" shares all bigrams with itself, "help" shares "he" and "el"
        let values: Vec<&str> = results.iter().map(|r| r.value.as_str()).collect();
        assert!(values.contains(&"hello"));
        assert!(values.contains(&"help"));
    }

    #[test]
    fn search_top_k() {
        let index = NgramIndex::build(&["hello", "help", "world", "held"], 2);
        let results = index.search_top_k("hello", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].value, "hello"); // most shared
    }

    #[test]
    fn empty_index() {
        let index = NgramIndex::build(&[], 2);
        assert!(index.is_empty());
        assert!(index.search("hello", 1).is_empty());
    }

    #[test]
    fn len() {
        let index = NgramIndex::build(&["a", "b", "c"], 2);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn high_threshold_filters() {
        let index = NgramIndex::build(&["hello", "world"], 2);
        let results = index.search("xyz", 1);
        assert!(results.is_empty());
    }
}
