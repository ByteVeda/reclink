<p align="center">
  <img src="https://raw.githubusercontent.com/ByteVeda/reclink/master/docs-site/static/img/icon.png" alt="reclink" width="120" height="120">
</p>

<h1 align="center">reclink</h1>

<p align="center">Blazing-fast fuzzy matching and record linkage library powered by Rust.</p>

<p align="center">
  <a href="https://pypi.org/project/reclink/"><img src="https://img.shields.io/pypi/v/reclink" alt="PyPI"></a>
  <a href="https://pypi.org/project/reclink/"><img src="https://img.shields.io/pypi/pyversions/reclink" alt="Python"></a>
  <a href="https://github.com/ByteVeda/reclink/actions/workflows/ci.yml"><img src="https://github.com/ByteVeda/reclink/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/ByteVeda/reclink/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ByteVeda/reclink" alt="License"></a>
</p>

## Features

- **21 string similarity metrics** — edit distance, token-based, subsequence, alignment, and hybrid metrics
- **10 phonetic algorithms** — Soundex, Metaphone, Double Metaphone, NYSIIS, Caverphone, Cologne, Beider-Morse, Phonex, MRA, Daitch-Mokotoff
- **Full record linkage pipeline** — blocking, comparison, classification, and clustering
- **11 blocking strategies** — exact, phonetic, sorted neighborhood, q-gram, LSH, canopy, trie, numeric, date, hybrid (union/intersection)
- **8 classifiers** — threshold, weighted, bands, Fellegi-Sunter with EM estimation, logistic regression, decision tree
- **5 clustering algorithms** — connected components, hierarchical, DBSCAN, OPTICS, incremental
- **7 index structures** — BK-tree, VP-tree, N-gram index, memory-mapped N-gram, MinHash LSH, Bloom filter, inverted index
- **DataFrame integration** — pandas and polars accessors, native Polars plugin
- **Parallel computation** — Rayon-powered `cdist` and pipeline execution
- **Scoring presets & composite scorer** — pre-tuned configs for name, address, and general-purpose matching
- **Extensible plugin system** — register custom metrics, blockers, comparators, classifiers, and preprocessors
- **WASM bindings** — run reclink in the browser

## Installation

```bash
pip install reclink
```

### Build from source

```bash
git clone https://github.com/ByteVeda/reclink.git
cd reclink
uv sync --extra dev
maturin develop --release
```

## Quick Start

### Record linkage pipeline

```python
import pandas as pd
from reclink.pipeline import ReclinkPipeline

df = pd.DataFrame({
    "id": ["1", "2", "3"],
    "first_name": ["Jon", "John", "Jane"],
    "last_name": ["Smith", "Smyth", "Doe"],
})

pipeline = (
    ReclinkPipeline.builder()
    .preprocess("first_name", ["fold_case", "strip_punctuation"])
    .preprocess("last_name", ["fold_case"])
    .block_phonetic("last_name", algorithm="soundex")
    .compare_string("first_name", metric="jaro_winkler")
    .compare_string("last_name", metric="jaro_winkler")
    .classify_threshold(0.85)
    .build()
)

matches = pipeline.dedup(df)
print(matches)
#    left_id right_id     score            scores
# 0       1        2  0.921...  [0.832..., 1.0...]
```

### Direct metric usage

```python
from reclink import jaro_winkler, soundex, cdist

jaro_winkler("Jon", "John")  # 0.93...
soundex("Smith") == soundex("Smyth")  # True
cdist(["Jon", "Jane"], ["John", "Janet"], scorer="jaro_winkler")  # 2x2 numpy array
```

## Documentation

Full documentation at [docs.byteveda.org/reclink](https://docs.byteveda.org/reclink/), including:

- **[API Reference](https://docs.byteveda.org/reclink/api/string-metrics)** — every metric, algorithm, and class
- **[Guides](https://docs.byteveda.org/reclink/guides/name-matching)** — pipelines, preprocessing, DataFrames, custom plugins
- **[Interactive Playground](https://docs.byteveda.org/reclink/playground/)** — try reclink in your browser (WASM-powered)
- **[Changelog](https://docs.byteveda.org/reclink/changelog)** — release history

## Performance

Pairwise comparison (10 string pairs, 500 iterations, microseconds per pair):

| Metric | reclink | rapidfuzz | jellyfish | vs rapidfuzz | vs jellyfish |
|--------|--------:|----------:|----------:|-------------:|-------------:|
| levenshtein | 0.55 | 0.18 | 1.28 | 3.0x slower | **2.3x faster** |
| jaro | 0.31 | 0.20 | 0.68 | 1.6x slower | **2.2x faster** |
| jaro_winkler | 0.31 | 0.20 | 0.68 | 1.5x slower | **2.2x faster** |
| damerau_levenshtein | 0.93 | 0.24 | 2.41 | 3.9x slower | **2.6x faster** |

Batch matching (1,000 candidates, 50 iterations, microseconds per candidate):

| Operation | reclink | rapidfuzz | thefuzz | vs rapidfuzz | vs thefuzz |
|-----------|--------:|----------:|--------:|-------------:|-----------:|
| match_batch (jaro_winkler) | 0.32 | 0.13 | 1.60 | 2.4x slower | **5.0x faster** |

Reproduce with `python benchmarks/compare.py` (requires `pip install rapidfuzz jellyfish thefuzz`).

## Development

```bash
# Setup
uv sync --extra dev
uv run pre-commit install
maturin develop --release

# Rust
cargo test --workspace
cargo clippy -- -D warnings
cargo fmt --check

# Python
uv run pytest tests/python/ -v
uv run ruff check py_src/ tests/
uv run ruff format --check py_src/ tests/
uv run mypy py_src/reclink/
```

## License

Apache-2.0
