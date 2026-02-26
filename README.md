# reclink

Blazing-fast fuzzy matching and record linkage library powered by Rust.

## Features

- **Rust core with Python bindings** via PyO3 — no pure-Python bottlenecks
- **10 string similarity metrics** implemented from scratch (Levenshtein, Jaro-Winkler, cosine, Jaccard, and more)
- **4 phonetic algorithms** — Soundex, Metaphone, Double Metaphone, NYSIIS
- **Full record linkage pipeline** — blocking, comparison, classification, and clustering
- **8 blocking strategies** — exact, phonetic, sorted neighborhood, q-gram, LSH, canopy, numeric, date
- **Fellegi-Sunter with EM estimation** — unsupervised probabilistic matching out of the box
- **DataFrame integration** — works with pandas, polars, or plain dicts
- **Parallel computation** — Rayon-powered `cdist` and pipeline execution
- **Evaluation & export** — precision/recall/F1, CSV/JSON export

## Installation

```bash
pip install reclink
```

### Build from source

```bash
git clone https://github.com/your-org/reclink.git
cd reclink
uv sync --extra dev
maturin develop --release
```

## Quick Start

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

## String Metrics

All metrics are implemented in Rust and callable directly from Python.

| Function | Signature | Returns |
|---|---|---|
| `levenshtein` | `(a, b)` | `int` — edit distance |
| `levenshtein_similarity` | `(a, b)` | `float` — normalized similarity [0, 1] |
| `damerau_levenshtein` | `(a, b)` | `int` — edit distance with transpositions |
| `damerau_levenshtein_similarity` | `(a, b)` | `float` — normalized similarity [0, 1] |
| `hamming` | `(a, b)` | `int` — positional differences (equal-length only) |
| `hamming_similarity` | `(a, b)` | `float` — normalized similarity [0, 1] |
| `jaro` | `(a, b)` | `float` — Jaro similarity [0, 1] |
| `jaro_winkler` | `(a, b, prefix_weight=0.1)` | `float` — Jaro-Winkler similarity [0, 1] |
| `cosine` | `(a, b, n=2)` | `float` — cosine similarity over character n-grams |
| `jaccard` | `(a, b)` | `float` — Jaccard index over whitespace tokens |
| `sorensen_dice` | `(a, b)` | `float` — Dice coefficient over character bigrams |

**Batch comparison:**

```python
from reclink import cdist

matrix = cdist(["Jon", "Jane"], ["John", "Janet"], scorer="jaro_winkler")
# Returns a 2x2 numpy array of similarity scores
```

`cdist` supports all similarity metrics and parallelizes across CPU cores automatically.

## Phonetic Algorithms

| Function | Signature | Returns |
|---|---|---|
| `soundex` | `(s)` | `str` — 4-character Soundex code |
| `metaphone` | `(s)` | `str` — variable-length Metaphone code |
| `double_metaphone` | `(s)` | `tuple[str, str]` — primary and alternate codes |
| `nysiis` | `(s)` | `str` — NYSIIS code (up to 6 characters) |

```python
from reclink import soundex, double_metaphone

soundex("Smith")         # "S530"
soundex("Smyth")         # "S530"
double_metaphone("John") # ("JN", "AN")
```

## Preprocessing

Individual operations and batch processing:

| Function | Description |
|---|---|
| `fold_case(s)` | Unicode case folding (lowercasing) |
| `normalize_whitespace(s)` | Collapse runs of whitespace to single space, trim |
| `strip_punctuation(s)` | Remove punctuation characters |
| `standardize_name(s)` | Normalize a personal name (fold case, strip punctuation, normalize whitespace) |
| `normalize_unicode(s, form="nfkc")` | Unicode normalization (NFC, NFKC, NFD, NFKD) |
| `ngram_tokenize(s, n=2)` | Character n-gram tokenization |
| `whitespace_tokenize(s)` | Split on whitespace |

**Batch API** — process many strings at once in Rust:

```python
from reclink import preprocess_batch, ngram_tokenize_batch

cleaned = preprocess_batch(
    ["  John Smith ", "JANE  DOE"],
    operations=["fold_case", "normalize_whitespace", "strip_punctuation"],
)
# ["john smith", "jane doe"]

tokens = ngram_tokenize_batch(["hello", "world"], n=2)
# [["he", "el", "ll", "lo"], ["wo", "or", "rl", "ld"]]
```

## Pipeline

The `ReclinkPipeline` uses a builder pattern to configure blocking, comparison, classification, and clustering stages.

### Per-field Preprocessing

Apply preprocessing operations before comparison:

```python
builder = (
    ReclinkPipeline.builder()
    .preprocess("name", ["fold_case", "normalize_whitespace", "strip_punctuation"])
    .preprocess("city", ["fold_case"])
)
```

### Blocking Strategies

Blocking reduces the number of candidate pairs by grouping records that are likely to match.

| Method | Description |
|---|---|
| `block_exact(field)` | Exact match on field value |
| `block_phonetic(field, algorithm="soundex")` | Match on phonetic encoding |
| `block_sorted_neighborhood(field, window=3)` | Sliding window over sorted values |
| `block_qgram(field, q=3, threshold=1)` | Minimum shared q-gram overlap |
| `block_lsh(field, num_hashes=100, num_bands=20)` | Locality-Sensitive Hashing (MinHash + banding) |
| `block_canopy(field, t_tight=0.9, t_loose=0.5, metric="jaro_winkler")` | Canopy clustering with two thresholds |
| `block_numeric(field, bucket_size=5.0)` | Numeric bucket ranges |
| `block_date(field, resolution="year")` | Date truncation (year/month/day) |

### Comparators

Compare field values to produce similarity scores:

| Method | Description |
|---|---|
| `compare_string(field, metric="jaro_winkler")` | String similarity using any metric |
| `compare_exact(field)` | Binary exact match (1.0 or 0.0) |
| `compare_numeric(field, max_diff=10.0)` | Numeric distance normalized by max_diff |
| `compare_date(field)` | Date similarity |
| `compare_phonetic(field, algorithm="soundex")` | Phonetic match (1.0 if same encoding, else 0.0) |

### Classifiers

Decide which candidate pairs are matches:

| Method | Description |
|---|---|
| `classify_threshold(threshold)` | Match if average score >= threshold |
| `classify_weighted(weights, threshold)` | Match if weighted sum >= threshold |
| `classify_fellegi_sunter(m_probs, u_probs, upper, lower)` | Probabilistic model with known parameters |
| `classify_fellegi_sunter_auto(max_iterations=100, ...)` | Fellegi-Sunter with EM-estimated parameters |

### Clustering

Group matched pairs into clusters:

| Method | Description |
|---|---|
| `cluster_connected_components()` | Transitive closure — all connected records form a cluster |
| `cluster_hierarchical(linkage="single", threshold=0.5)` | Agglomerative clustering (single/complete/average linkage) |

### Running the Pipeline

```python
# Deduplication — find matches within a single dataset
matches = pipeline.dedup(df, id_column="id")

# Deduplication with clustering — group duplicate records
clusters = pipeline.dedup_cluster(df, id_column="id")

# Record linkage — match across two datasets
matches = pipeline.link(df_left, df_right, id_column="id")
```

All methods accept pandas DataFrames, polars DataFrames, or lists of dicts, and return the same type as the input.

## Evaluation

Measure linkage quality against ground truth:

```python
from reclink.evaluation import precision, recall, f1_score, confusion_matrix, pairs_from_results

predicted = pairs_from_results(matches)
truth = {("1", "2"), ("3", "4")}

precision(predicted, truth)        # fraction of predicted that are correct
recall(predicted, truth)           # fraction of true matches found
f1_score(predicted, truth)         # harmonic mean of precision and recall
confusion_matrix(predicted, truth) # {"tp": ..., "fp": ..., "fn": ...}
```

## Export

Write results to CSV or JSON:

```python
from reclink.export import (
    export_matches_csv,
    export_matches_json,
    export_clusters_csv,
    export_clusters_json,
)

export_matches_csv(matches, "matches.csv")
export_matches_json(matches, "matches.json")
export_clusters_csv(clusters, "clusters.csv")
export_clusters_json(clusters, "clusters.json")
```

## Performance

reclink is designed for speed:

- **Rust core** — all string metrics, phonetic algorithms, and pipeline logic run in compiled Rust
- **Rayon parallelism** — `cdist` and pipeline stages parallelize across CPU cores
- **Enum dispatch** — metrics use enum dispatch instead of dynamic dispatch, avoiding vtable overhead
- **Zero-copy where possible** — PyO3 bindings minimize data copying between Python and Rust

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
