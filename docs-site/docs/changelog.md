---
title: Changelog
sidebar_label: Changelog
slug: /changelog
---

# Changelog

All notable changes to reclink will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## 0.1.1

### Fixed

- PyPI sdist now includes LICENSE file at package root
- PyPI README logo renders correctly (absolute URL)
- Type stubs: `Scorer` and `CompositePreset` accept `str` for custom metric support
- `register_metric` has proper `@overload` signatures for all calling conventions
- Clippy and ruff lint errors resolved across codebase
- Mypy runs on test files with correct `--python-executable` for numpy resolution
- WASM playground loads correctly under `/reclink/` base path
- Internal doc links updated for `routeBasePath: '/'`

### Added

- Pre-commit hooks: cargo fmt, clippy, ruff check, ruff format, mypy
- CI workflow with lint, rust-test, and cross-platform Python test matrix
- GitHub Pages docs workflow with WASM build
- PyPI release workflow with trusted publishing (OIDC) for 6 platform targets
- PR cache cleanup workflow
- Changelog page in docs site
- PyPI metadata: badges, classifiers, keywords, project URLs

### Changed

- Docs served at `docs.byteveda.org/reclink/` via org-level GitHub Pages
- Removed `/docs` prefix from doc routes (redundant with `docs.` subdomain)
- Updated crate dependencies: `unicode-normalization` 0.1.25, `ahash` 0.8.12, `thiserror` 2.0, `regex` 1.12, `memmap2` 0.9.10, `crossbeam-channel` 0.5.15, `wasm-bindgen` 0.2.114, `serde-wasm-bindgen` 0.6.5, `js-sys` 0.3.91
- Apache-2.0 license consistent across all crates

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
