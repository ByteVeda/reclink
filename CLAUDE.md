# reclink

Rust-powered fuzzy matching & record linkage library with Python bindings.

## Directory Layout

```
├── src/lib.rs                   # PyO3 bindings (cdylib "_core")
├── crates/reclink-core/src/     # Pure Rust library
│   ├── metrics/                 # String distance & similarity metrics
│   ├── phonetic/                # Phonetic algorithms
│   ├── blocking/                # Blocking strategies
│   ├── compare/                 # Field comparators
│   ├── classify/                # Match classifiers
│   ├── cluster/                 # Clustering algorithms
│   ├── preprocess/              # Text preprocessing
│   ├── pipeline.rs              # Record linkage pipeline
│   ├── record.rs                # Record type
│   └── error.rs                 # Error types
├── crates/reclink-core/benches/ # Criterion benchmarks
├── py_src/reclink/              # Python package (wrappers, types, stubs)
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
cargo test --workspace            # Run all Rust tests (97 tests)
cargo clippy -- -D warnings      # Lint (must be warning-free)
cargo fmt --check                # Format check
cargo bench -p reclink-core      # Run Criterion benchmarks

# Python
uv run pytest tests/python/ -v   # Run Python tests (34 tests)
uv run ruff check py_src/ tests/ # Lint
uv run ruff format --check py_src/ tests/  # Format check
uv run mypy py_src/reclink/      # Type check (strict mode)
```

## Architecture & Conventions

- **Workspace**: root crate (PyO3 cdylib) + `crates/reclink-core` (pure Rust lib)
- All string metrics implemented from scratch (no external metric dependencies)
- Enum dispatch for metrics (avoids vtable overhead)
- Rayon parallelism in pipeline and cdist
- Trait-based design: `DistanceMetric`, `SimilarityMetric`, `BlockingStrategy`, `FieldComparator`, `Classifier`
- Builder pattern for pipeline construction
- Python wrappers use NumPy-style docstrings
- PEP 561 compliant: `py.typed` marker + `_core.pyi` stub

## Code Quality Gates

- **Rust**: `cargo clippy -- -D warnings` clean, `cargo fmt` clean, all tests pass
- **Python**: ruff (E/W/F/I/N/UP/B/SIM/TCH/RUF), mypy strict, all tests pass
- **Line length**: 100 (Python, configured in pyproject.toml)

## Key Dependencies

**Rust** (root): pyo3 0.24, numpy 0.24, rayon 1.11, ahash 0.8
**Rust** (reclink-core): rayon 1.11, unicode-normalization 0.1, ahash 0.8, thiserror 2
**Python**: numpy (required), pandas/polars (optional)
**Build**: maturin >=1.7, Python >=3.12
