---
title: Changelog
sidebar_label: Changelog
slug: /changelog
---

# Changelog

All notable changes to reclink will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## 0.1.0

Initial release.

### Added

- **21 string metrics**: Levenshtein, Damerau-Levenshtein, Hamming, Jaro, Jaro-Winkler, Cosine, Jaccard, Sorensen-Dice, Weighted Levenshtein, Token Sort, Token Set, Partial Ratio, LCS, Longest Common Substring, N-gram Similarity, Smith-Waterman, Phonetic Hybrid, Ratcliff-Obershelp, Needleman-Wunsch, Gotoh, Monge-Elkan
- **10 phonetic algorithms**: Soundex, Metaphone, Double Metaphone, NYSIIS, Caverphone, Cologne Phonetic, Beider-Morse, Phonex, MRA, Daitch-Mokotoff
- **11 blocking strategies**: Exact, Phonetic, Sorted Neighborhood, Trie, Q-gram, Numeric, Date, LSH, Canopy, Hybrid (union/intersection modes), Custom
- **8 classifiers**: Threshold, Weighted, Threshold Bands, Weighted Bands, Fellegi-Sunter, Fellegi-Sunter Auto (EM), Logistic Regression, Decision Tree
- **5 clustering algorithms**: Connected Components, Hierarchical, DBSCAN, OPTICS, Incremental
- **7 index structures**: BK-tree, VP-tree, N-gram Index, Memory-mapped N-gram Index, MinHash LSH, Bloom Filter, Inverted Index
- **Full record linkage pipeline** with builder pattern: preprocessing, blocking, comparison, classification, clustering
- **Batch operations**: `cdist`, `match_best`, `match_batch` with Rayon parallelism
- **Composite scoring** with presets (name matching, address matching, general purpose)
- **Streaming matchers** with backpressure support
- **Custom metric/blocker/comparator/classifier registration** with decorator syntax
- **Pandas and Polars integration**: DataFrame accessors, fuzzy join, Polars expression plugin
- **Text preprocessing**: case folding, whitespace normalization, diacritics stripping, stop word removal, abbreviation expansion, domain-specific cleaners (name, address, company, email, URL), transliteration (Cyrillic, Greek, Arabic, Hebrew, Devanagari, Hangul)
- **Arrow-compatible batch operations** for zero-copy interop
- **TF-IDF matcher** for corpus-aware similarity
- **EM parameter estimation** for Fellegi-Sunter models
- **Evaluation utilities**: precision, recall, F1, confusion matrix
- **Export**: CSV, JSON, NetworkX graph
- **CLI**: `reclink compare`, `reclink batch`, `reclink pipeline`
- **WASM bindings** for browser-based playground
- **Interactive docs site** with live WASM-powered playground
