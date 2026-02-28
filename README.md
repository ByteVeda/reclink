# reclink

Blazing-fast fuzzy matching and record linkage library powered by Rust.

## Features

- **20+ string similarity metrics** — edit distance, token-based, subsequence, alignment, and more
- **7 phonetic algorithms** — Soundex, Metaphone, Double Metaphone, NYSIIS, Caverphone, Cologne, Beider-Morse
- **Full record linkage pipeline** — blocking, comparison, classification, and clustering
- **9 blocking strategies** — exact, phonetic, sorted neighborhood, q-gram, LSH, canopy, trie, numeric, date
- **Fellegi-Sunter with EM estimation** — unsupervised probabilistic matching out of the box
- **Index structures** — BK-tree, VP-tree, N-gram index, memory-mapped N-gram, and MinHash/LSH
- **DataFrame integration** — pandas and polars accessors, native Polars plugin
- **Parallel computation** — Rayon-powered `cdist` and pipeline execution
- **Scoring presets & composite scorer** — pre-tuned configs for name, address, and general-purpose matching
- **Extensible plugin system** — register custom metrics, blockers, comparators, classifiers, and preprocessors

## Installation

```bash
pip install reclink
```

### Build from source

```bash
git clone https://github.com/pratyush618/reclink.git
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
from reclink import jaro_winkler_similarity, soundex, cdist

jaro_winkler_similarity("Jon", "John")  # 0.93...
soundex("Smith") == soundex("Smyth")    # True
cdist(["Jon", "Jane"], ["John", "Janet"], scorer="jaro_winkler")  # 2x2 numpy array
```

## Documentation

Full documentation is available at [reclink.dev](https://reclink.dev), including:

- **API Reference** — every metric, algorithm, and class
- **Guides** — pipelines, preprocessing, DataFrames, custom plugins, and more
- **Interactive Playground** — try reclink directly in your browser

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
maturin develop --release

# Rust
cargo build
cargo test --workspace
cargo clippy -- -D warnings
cargo fmt --check
cargo bench -p reclink-core

# Python
uv run pytest tests/python/ -v
uv run ruff check py_src/ tests/
uv run ruff format --check py_src/ tests/
uv run mypy py_src/reclink/
```

## License

Apache-2.0
