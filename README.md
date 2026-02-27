# reclink

Blazing-fast fuzzy matching and record linkage library powered by Rust.

## Features

- **Rust core with Python bindings** via PyO3 — no pure-Python bottlenecks
- **20+ string similarity metrics** implemented from scratch (edit distance, token-based, subsequence, alignment, and more)
- **6 phonetic algorithms** — Soundex, Metaphone, Double Metaphone, NYSIIS, Caverphone, Cologne
- **Domain preprocessors** — `clean_name`, `clean_address`, `clean_company`, email/URL normalization, synonym expansion
- **Scoring presets** — pre-tuned `CompositeScorer` configs for name matching, address matching, and general-purpose use
- **Index structures** — BK-tree, VP-tree, and N-gram index for sub-linear nearest-neighbor search
- **Streaming matcher** — lazy/chunked matching over iterators without materializing the full dataset
- **TF-IDF matcher** — corpus-aware similarity scoring
- **Full record linkage pipeline** — blocking, comparison, classification, and clustering
- **8 blocking strategies** — exact, phonetic, sorted neighborhood, q-gram, LSH, canopy, numeric, date
- **Fellegi-Sunter with EM estimation** — unsupervised probabilistic matching out of the box
- **DataFrame integration** — pandas and polars accessors for fuzzy merge, phonetic encoding, and deduplication
- **Parallel computation** — Rayon-powered `cdist` and pipeline execution
- **Evaluation & export** — precision/recall/F1, CSV/JSON export

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

### Scoring presets

```python
from reclink import CompositeScorer
from reclink.presets import name_matching

scorer = name_matching()  # pre-tuned for person names
scorer.similarity("Jon Smith", "John Smyth")  # 0.89...

# Or construct directly
scorer = CompositeScorer.preset("address_matching")
scorer.match_best("123 Main St", ["123 Main Street", "456 Oak Ave"])
# ("123 Main Street", 0.94..., 0)
```

## String Metrics

All metrics are implemented in Rust and callable directly from Python.

### Edit distance

| Function | Signature | Returns |
|---|---|---|
| `levenshtein` | `(a, b)` | `int` — edit distance |
| `levenshtein_similarity` | `(a, b)` | `float` — normalized similarity [0, 1] |
| `damerau_levenshtein` | `(a, b)` | `int` — edit distance with transpositions |
| `damerau_levenshtein_similarity` | `(a, b)` | `float` — normalized similarity [0, 1] |
| `hamming` | `(a, b)` | `int` — positional differences (equal-length only) |
| `hamming_similarity` | `(a, b)` | `float` — normalized similarity [0, 1] |
| `weighted_levenshtein` | `(a, b, insert_cost, delete_cost, substitute_cost, transpose_cost)` | `float` — weighted edit distance with configurable costs |
| `weighted_levenshtein_similarity` | `(a, b, ...)` | `float` — normalized weighted similarity [0, 1] |

### Similarity

| Function | Signature | Returns |
|---|---|---|
| `jaro` | `(a, b)` | `float` — Jaro similarity [0, 1] |
| `jaro_winkler` | `(a, b, prefix_weight=0.1)` | `float` — Jaro-Winkler similarity [0, 1] |

### Token-based

| Function | Signature | Returns |
|---|---|---|
| `cosine` | `(a, b, n=2)` | `float` — cosine similarity over character n-grams |
| `jaccard` | `(a, b)` | `float` — Jaccard index over whitespace tokens |
| `sorensen_dice` | `(a, b)` | `float` — Dice coefficient over character bigrams |
| `token_sort_ratio` | `(a, b)` | `float` — similarity after sorting tokens alphabetically |
| `token_set_ratio` | `(a, b)` | `float` — similarity using set intersection/remainder logic |
| `partial_ratio` | `(a, b)` | `float` — best substring match similarity |
| `ngram_similarity` | `(a, b, n=2)` | `float` — Jaccard coefficient over character n-gram sets |

### Subsequence

| Function | Signature | Returns |
|---|---|---|
| `lcs_length` | `(a, b)` | `int` — length of longest common subsequence |
| `lcs_similarity` | `(a, b)` | `float` — normalized LCS similarity [0, 1] |
| `longest_common_substring_length` | `(a, b)` | `int` — length of longest contiguous common substring |
| `longest_common_substring_similarity` | `(a, b)` | `float` — normalized substring similarity [0, 1] |

### Alignment

| Function | Signature | Returns |
|---|---|---|
| `smith_waterman` | `(a, b, match_score, mismatch_penalty, gap_penalty)` | `float` — raw local alignment score |
| `smith_waterman_similarity` | `(a, b)` | `float` — normalized alignment similarity [0, 1] |

### Hybrid

| Function | Signature | Returns |
|---|---|---|
| `phonetic_hybrid` | `(a, b, phonetic="soundex", metric="jaro_winkler", phonetic_weight=0.3)` | `float` — weighted phonetic + edit distance similarity |

### Early termination

| Function | Signature | Returns |
|---|---|---|
| `levenshtein_threshold` | `(a, b, max_distance)` | `int \| None` — distance if within threshold, else None |
| `damerau_levenshtein_threshold` | `(a, b, max_distance)` | `int \| None` — distance if within threshold, else None |

### Batch comparison

```python
from reclink import cdist

matrix = cdist(["Jon", "Jane"], ["John", "Janet"], scorer="jaro_winkler")
# Returns a 2x2 numpy array of similarity scores
```

`cdist` supports all similarity metrics and parallelizes across CPU cores automatically.

### Match best & batch

```python
from reclink import match_best, match_batch

# Find the single best match
match_best("hello", ["hallo", "world", "help"])
# ("hallo", 0.93..., 0)

# Find all matches above a threshold
match_batch("hello", ["hallo", "world", "help"], threshold=0.5, limit=2)
# [("hallo", 0.93..., 0), ("help", 0.73..., 2)]
```

## Phonetic Algorithms

| Function | Signature | Returns |
|---|---|---|
| `soundex` | `(s)` | `str` — 4-character Soundex code |
| `metaphone` | `(s)` | `str` — variable-length Metaphone code |
| `double_metaphone` | `(s)` | `tuple[str, str]` — primary and alternate codes |
| `nysiis` | `(s)` | `str` — NYSIIS code (up to 6 characters) |
| `caverphone` | `(s)` | `str` — Caverphone code |
| `cologne_phonetic` | `(s)` | `str` — Cologne phonetic code |

```python
from reclink import soundex, double_metaphone, caverphone, cologne_phonetic

soundex("Smith")           # "S530"
soundex("Smyth")           # "S530"
double_metaphone("John")   # ("JN", "AN")
caverphone("Thompson")     # "TMSN111111"
cologne_phonetic("Müller") # "657"
```

## Preprocessing

### Basic text operations

| Function | Description |
|---|---|
| `fold_case(s)` | Unicode case folding (lowercasing) |
| `normalize_whitespace(s)` | Collapse runs of whitespace to single space, trim |
| `strip_punctuation(s)` | Remove punctuation characters |
| `standardize_name(s)` | Normalize a personal name (fold case, strip punctuation, normalize whitespace) |
| `normalize_unicode(s, form="nfkc")` | Unicode normalization (NFC, NFKC, NFD, NFKD) |
| `strip_diacritics(s)` | Remove diacritical marks (accents) |
| `remove_stop_words(s)` | Remove common stop words |
| `expand_abbreviations(s)` | Expand common abbreviations |
| `regex_replace(s, pattern, replacement)` | Replace via regular expression |

### Tokenization

| Function | Description |
|---|---|
| `ngram_tokenize(s, n=2)` | Character n-gram tokenization |
| `whitespace_tokenize(s)` | Split on whitespace |

### Domain preprocessors

Clean and normalize domain-specific data before matching:

```python
from reclink import clean_name, clean_address, clean_company

clean_name("  DR. John  Smith Jr.  ")     # "john smith"
clean_address("123 N. Main St., Apt #4")  # "123 n main st apt 4"
clean_company("The Acme Corp., Inc.")      # "acme"
```

### Email and URL normalization

```python
from reclink import normalize_email, normalize_url

normalize_email("John.Doe+tag@Gmail.COM")  # "johndoe@gmail.com"
normalize_url("HTTP://WWW.Example.COM/path/")  # "example.com/path"
```

### Synonym expansion

```python
from reclink import synonym_expand

table = {"st": "street", "ave": "avenue", "dr": "drive"}
synonym_expand("123 main st", table)  # "123 main street"
```

### Batch API

Process many strings at once in Rust:

```python
from reclink import preprocess_batch, ngram_tokenize_batch, whitespace_tokenize_batch

cleaned = preprocess_batch(
    ["  John Smith ", "JANE  DOE"],
    operations=["fold_case", "normalize_whitespace", "strip_punctuation"],
)
# ["john smith", "jane doe"]

tokens = ngram_tokenize_batch(["hello", "world"], n=2)
# [["he", "el", "ll", "lo"], ["wo", "or", "rl", "ld"]]

words = whitespace_tokenize_batch(["hello world", "foo bar"])
# [["hello", "world"], ["foo", "bar"]]
```

## Scoring Presets

Pre-tuned `CompositeScorer` configurations for common use cases:

```python
from reclink.presets import name_matching, address_matching, general_purpose

scorer = name_matching()
# Weights: jaro_winkler=0.5, token_sort=0.3, phonetic_hybrid=0.2

scorer = address_matching()
# Weights: token_set=0.5, jaccard=0.3, levenshtein=0.2

scorer = general_purpose()
# Weights: jaro_winkler=0.4, cosine=0.4, token_sort=0.2
```

You can also load presets by name:

```python
from reclink import CompositeScorer

scorer = CompositeScorer.preset("name_matching")
```

## Composite Scorer

Build custom weighted scorers from any combination of metrics:

```python
from reclink import CompositeScorer

scorer = CompositeScorer([
    ("jaro_winkler", 0.6),
    ("token_sort", 0.4),
])

scorer.similarity("Jon Smith", "John Smyth")  # weighted blend
scorer.match_best("Jon", ["John", "Jane", "James"])
scorer.match_batch("Jon", ["John", "Jane", "James"], threshold=0.7)
```

## TF-IDF Matcher

Corpus-aware similarity that down-weights common tokens:

```python
from reclink import TfIdfMatcher

corpus = ["John Smith", "Jane Doe", "John Doe", "Jane Smith"]
matcher = TfIdfMatcher.fit(corpus)

matcher.similarity("John Smith", "John Doe")  # lower (shared "John")
matcher.match_batch("John Smith", corpus, threshold=0.1)
```

## Index Structures

Persistent, sub-linear nearest-neighbor search structures:

### BK-tree

For edit-distance metrics (Levenshtein, Damerau-Levenshtein, Hamming):

```python
from reclink import BkTree

tree = BkTree.build(["smith", "smyth", "john", "jane"], metric="levenshtein")
tree.find_within("smith", max_distance=1)  # [("smith", 0, 0), ("smyth", 1, 1)]
tree.find_nearest("jon", k=2)              # [("john", 1, 2), ("jane", 2, 3)]

tree.save("names.bk")
tree = BkTree.load("names.bk")
```

### VP-tree

For any metric (similarity or distance):

```python
from reclink import VpTree

tree = VpTree.build(["smith", "smyth", "john", "jane"], metric="jaro_winkler")
tree.find_nearest("jon", k=2)
tree.find_within("smith", max_distance=0.3)

tree.save("names.vp")
tree = VpTree.load("names.vp")
```

### N-gram index

For fast approximate matching via shared n-gram overlap:

```python
from reclink import NgramIndex

index = NgramIndex.build(["smith", "smyth", "john", "jane"], n=2)
index.search("smith", threshold=2)  # minimum shared bigrams
index.search_top_k("smith", k=2)

index.save("names.ngram")
index = NgramIndex.load("names.ngram")
```

## Streaming

Match against large or unbounded candidate sets lazily:

```python
from reclink.streaming import match_stream

# Works with any iterable — files, generators, database cursors
candidates = ("candidate_" + str(i) for i in range(1_000_000))

for matched_string, score, index in match_stream("target", candidates, threshold=0.8):
    print(f"{matched_string} (score={score:.3f}, index={index})")
```

Candidates are processed in configurable chunks (default 1000) for efficiency.

For lower-level control, use `StreamingMatcher` directly:

```python
from reclink import StreamingMatcher

matcher = StreamingMatcher("target", scorer="jaro_winkler", threshold=0.8)
matcher.score("candidate")              # float | None
matcher.score_chunk(["a", "b", "c"])    # [(index, score), ...]
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

You can also estimate Fellegi-Sunter parameters independently via `estimate_fellegi_sunter(vectors)`, which returns an `EmResult` with `m_probs`, `u_probs`, `p_match`, `iterations`, and `converged`.

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

All methods accept pandas DataFrames, polars DataFrames, or lists of dicts, and return the same type as the input. Match results are returned as `MatchResult` objects with `left_id`, `right_id`, `score`, and `scores` attributes.

## DataFrame Integration

### Pandas

```python
import pandas as pd

df = pd.DataFrame({"name": ["Jon Smith", "Jane Doe", "John Smyth"]})
candidates = ["John Smith", "Janet Doe"]

# Series accessor — match each value against candidates
df["name"].reclink.match_best(candidates, scorer="jaro_winkler", threshold=0.7)

# Series accessor — phonetic encoding
df["name"].reclink.phonetic(algorithm="soundex")

# Series accessor — find duplicate groups
df["name"].reclink.deduplicate(threshold=0.85)
# [[0, 2]]  — rows 0 and 2 are duplicates

# DataFrame accessor — fuzzy merge
right = pd.DataFrame({"company": ["Acme Inc", "Globex"], "revenue": [100, 200]})
left = pd.DataFrame({"firm": ["ACME", "globex corp"]})
left.reclink.fuzzy_merge(right, left_on="firm", right_on="company", threshold=0.6)
```

### Polars

```python
import polars as pl

s = pl.Series("name", ["Jon Smith", "Jane Doe", "John Smyth"])

# Match best
s.reclink.match_best(["John Smith", "Janet Doe"], scorer="jaro_winkler")

# Phonetic encoding
s.reclink.phonetic(algorithm="soundex")
```

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

## Explain

Understand why two strings match (or don't) across multiple algorithms:

```python
from reclink import explain

explain("Jon Smith", "John Smyth")
# {"levenshtein": 3, "jaro_winkler": 0.832, "soundex": 1.0, ...}

explain("Jon Smith", "John Smyth", algorithms=["jaro_winkler", "token_sort_ratio"])
# {"jaro_winkler": 0.832, "token_sort_ratio": 0.87}
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
