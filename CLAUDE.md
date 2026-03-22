# reclink

Rust-powered fuzzy matching & record linkage library with Python and WASM bindings.

## Directory Layout

```
├── crates/reclink-core/src/     # Pure Rust library
│   ├── metrics/                 # 21 string distance & similarity metrics
│   ├── phonetic/                # 10 phonetic algorithms
│   ├── blocking/                # 11 blocking strategies (incl. hybrid)
│   ├── compare/                 # Field comparators
│   ├── classify/                # 8 match classifiers (incl. logistic reg, decision tree)
│   ├── cluster/                 # 5 clustering algorithms (incl. DBSCAN, OPTICS)
│   ├── index/                   # 7 index structures (BK-tree, VP-tree, Bloom, etc.)
│   ├── preprocess/              # Text preprocessing
│   ├── pipeline.rs              # Record linkage pipeline
│   ├── record.rs                # Record type
│   └── error.rs                 # Error types
├── crates/reclink-core/benches/ # Criterion benchmarks
├── crates/reclink-py/src/       # PyO3 bindings (cdylib "_core")
│   ├── lib.rs                   # Module registration
│   ├── metrics.rs               # String metrics bindings
│   ├── phonetic.rs              # Phonetic algorithm bindings
│   ├── preprocess.rs            # Preprocessing bindings
│   ├── index.rs                 # Index structure bindings
│   ├── explain.rs               # Explain bindings
│   ├── scoring.rs               # Scoring/streaming bindings
│   ├── plugins.rs               # Custom plugin registration
│   ├── arrow_interop.rs         # Arrow batch operations
│   ├── parsers.rs               # Config string parsers
│   ├── polars_plugin.rs         # Polars expression plugin
│   └── pipeline/                # Pipeline bindings
├── crates/reclink-wasm/src/     # WASM bindings (wasm-bindgen)
│   ├── lib.rs                   # Module entry point
│   ├── metrics.rs               # String metrics for JS
│   ├── phonetic.rs              # Phonetic encoders for JS
│   ├── preprocess.rs            # Preprocessing for JS
│   ├── batch.rs                 # Batch matching for JS
│   └── index.rs                 # Index structures for JS
├── py_src/reclink/              # Python package (wrappers, types, stubs)
├── docs-site/                   # Docusaurus docs + WASM playground
├── tests/python/                # pytest tests
└── tests/rust/                  # Rust integration tests
```

## Build & Dev Commands

```bash
# Setup
uv sync --extra dev              # Install all dev dependencies
maturin develop --release        # Build extension in-place

# Rust
cargo build                      # Build all crates
cargo test --workspace            # Run all Rust tests (527 tests)
cargo clippy -- -D warnings      # Lint (must be warning-free)
cargo fmt --check                # Format check
cargo bench -p reclink-core      # Run Criterion benchmarks

# Python
uv run pytest tests/python/ -v   # Run Python tests (432 tests)
uv run ruff check py_src/ tests/ # Lint
uv run ruff format --check py_src/ tests/  # Format check
uv run mypy py_src/reclink/      # Type check (strict mode)

# WASM
wasm-pack build crates/reclink-wasm --target web --release --out-dir ../../docs-site/static/wasm
```

## Architecture & Conventions

- **Workspace**: `crates/reclink-core` (pure Rust) + `crates/reclink-py` (PyO3 cdylib) + `crates/reclink-wasm` (WASM cdylib)
- All string metrics implemented from scratch (no external metric dependencies)
- Enum dispatch for metrics (avoids vtable overhead)
- Rayon parallelism in pipeline and cdist (feature-gated: `parallel` feature in reclink-core)
- Trait-based design: `DistanceMetric`, `SimilarityMetric`, `BlockingStrategy`, `FieldComparator`, `Classifier`, `PhoneticEncoder`
- Builder pattern for pipeline construction
- Python wrappers use NumPy-style docstrings
- PEP 561 compliant: `py.typed` marker + `_core.pyi` stub

## Code Quality Gates

- **Rust**: `cargo clippy -- -D warnings` clean, `cargo fmt` clean, all tests pass
- **Python**: ruff (E/W/F/I/N/UP/B/SIM/TCH/RUF), mypy strict, all tests pass
- **Line length**: 100 (Python, configured in pyproject.toml)

## Key Dependencies

**Rust** (reclink-py): pyo3 0.24, numpy 0.24, rayon 1.11, ahash 0.8
**Rust** (reclink-core): rayon 1.11 (optional), unicode-normalization 0.1, ahash 0.8, thiserror 2
**Rust** (reclink-wasm): wasm-bindgen 0.2, serde-wasm-bindgen 0.6, js-sys 0.3
**Python**: numpy (required), pandas/polars (optional)
**Build**: maturin >=1.7, Python >=3.12, wasm-pack 0.14+
