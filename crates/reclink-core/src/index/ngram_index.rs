//! Inverted n-gram index for fast approximate string matching.
//!
//! Maps each character n-gram to the set of strings containing it.
//! At query time, counts how many n-grams the query shares with each candidate
//! and returns those above a threshold.

use ahash::{AHashMap, AHashSet};
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
    #[serde(default)]
    deleted: AHashSet<usize>,
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
            deleted: AHashSet::new(),
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

    /// Insert a new string and return its assigned index.
    pub fn insert_new(&mut self, s: &str) -> usize {
        let index = self.strings.len();
        self.strings.push(s.to_string());
        let ngrams = ngram_tokenize(s, self.n);
        for ng in ngrams {
            self.index.entry(ng).or_default().push(index);
        }
        index
    }

    /// Soft-delete a string by index. Returns `true` if the index was valid
    /// and not already deleted.
    pub fn remove(&mut self, index: usize) -> bool {
        if index >= self.strings.len() || self.deleted.contains(&index) {
            return false;
        }
        self.deleted.insert(index);
        true
    }

    /// Returns `true` if the index is valid and not deleted.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        index < self.strings.len() && !self.deleted.contains(&index)
    }

    /// Returns the number of active (non-deleted) strings in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strings.len() - self.deleted.len()
    }

    /// Returns whether the index has no active strings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimates the heap memory usage of this index in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        // strings vec
        for s in &self.strings {
            bytes += std::mem::size_of::<String>() + s.capacity();
        }
        // inverted index map
        for (key, postings) in &self.index {
            bytes += std::mem::size_of::<String>() + key.capacity();
            bytes += std::mem::size_of::<Vec<usize>>()
                + postings.capacity() * std::mem::size_of::<usize>();
        }
        // deleted set
        bytes += self.deleted.capacity() * std::mem::size_of::<usize>();
        bytes
    }

    fn count_shared(&self, query_ngrams: &[String]) -> Vec<(usize, usize)> {
        let mut counts: AHashMap<usize, usize> = AHashMap::new();
        for ng in query_ngrams {
            if let Some(indices) = self.index.get(ng) {
                for &idx in indices {
                    if !self.deleted.contains(&idx) {
                        *counts.entry(idx).or_insert(0) += 1;
                    }
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

    #[test]
    fn insert_after_build() {
        let mut index = NgramIndex::build(&["hello", "world"], 2);
        let idx = index.insert_new("help");
        assert_eq!(idx, 2);
        assert_eq!(index.len(), 3);
        let results = index.search("help", 2);
        let indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(indices.contains(&2));
    }

    #[test]
    fn remove_excludes_from_search() {
        let mut index = NgramIndex::build(&["hello", "help", "world"], 2);
        assert!(index.contains(1));
        assert!(index.remove(1)); // remove "help"
        assert!(!index.contains(1));
        assert_eq!(index.len(), 2);

        let results = index.search("help", 3);
        let indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(!indices.contains(&1));
    }

    #[test]
    fn remove_idempotent() {
        let mut index = NgramIndex::build(&["hello"], 2);
        assert!(index.remove(0));
        assert!(!index.remove(0));
        assert!(!index.remove(99));
    }
}
