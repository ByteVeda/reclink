"""Built-in benchmarking utilities for reclink.

Provides functions to benchmark individual string metrics and full
record-linkage pipelines on user-supplied data.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from reclink._core import (
    cosine,
    damerau_levenshtein_similarity,
    hamming_similarity,
    jaccard,
    jaro,
    jaro_winkler,
    lcs_similarity,
    levenshtein_similarity,
    longest_common_substring_similarity,
    ngram_similarity,
    partial_ratio,
    smith_waterman_similarity,
    sorensen_dice,
    token_set_ratio,
    token_sort_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from reclink.pipeline import ReclinkPipeline

_DEFAULT_METRICS: dict[str, Callable[[str, str], float]] = {
    "levenshtein": levenshtein_similarity,
    "damerau_levenshtein": damerau_levenshtein_similarity,
    "jaro": jaro,
    "jaro_winkler": jaro_winkler,
    "cosine": cosine,
    "jaccard": jaccard,
    "sorensen_dice": sorensen_dice,
    "token_sort": token_sort_ratio,
    "token_set": token_set_ratio,
    "partial_ratio": partial_ratio,
    "lcs": lcs_similarity,
    "longest_common_substring": longest_common_substring_similarity,
    "ngram": ngram_similarity,
    "smith_waterman": smith_waterman_similarity,
}


def benchmark_metrics(
    pairs: Sequence[tuple[str, str]],
    metrics: Sequence[str] | None = None,
    n: int = 1000,
) -> dict[str, Any]:
    """Benchmark string metrics on user-supplied string pairs.

    Parameters
    ----------
    pairs : sequence of (str, str)
        String pairs to compare.
    metrics : sequence of str or None
        Metric names to benchmark. If None, benchmarks all built-in
        similarity metrics.
    n : int
        Number of iterations per metric (each iteration runs all pairs).

    Returns
    -------
    dict
        Keys: ``results`` (list of per-metric dicts with ``metric``,
        ``total_ns``, ``per_pair_ns``, ``pairs_per_sec``),
        ``n_iterations``, ``n_pairs``.
    """
    pairs = list(pairs)
    if not pairs:
        return {"results": [], "n_iterations": n, "n_pairs": 0}

    if metrics is None:
        selected = _DEFAULT_METRICS
    else:
        selected = {}
        for name in metrics:
            if name not in _DEFAULT_METRICS:
                msg = f"Unknown metric: {name!r}. Available: {', '.join(sorted(_DEFAULT_METRICS))}"
                raise ValueError(msg)
            selected[name] = _DEFAULT_METRICS[name]

    # Warm up
    for fn in selected.values():
        for a, b in pairs:
            fn(a, b)

    results = []
    for name, fn in selected.items():
        start = time.perf_counter_ns()
        for _ in range(n):
            for a, b in pairs:
                fn(a, b)
        elapsed_ns = time.perf_counter_ns() - start

        total_calls = n * len(pairs)
        per_pair_ns = elapsed_ns / total_calls if total_calls > 0 else 0
        pairs_per_sec = 1_000_000_000 / per_pair_ns if per_pair_ns > 0 else float("inf")

        results.append(
            {
                "metric": name,
                "total_ns": elapsed_ns,
                "per_pair_ns": round(per_pair_ns, 1),
                "pairs_per_sec": round(pairs_per_sec),
            }
        )

    results.sort(key=lambda r: float(r["per_pair_ns"]))  # type: ignore[arg-type]

    return {
        "results": results,
        "n_iterations": n,
        "n_pairs": len(pairs),
    }


def benchmark_pipeline(
    pipeline: ReclinkPipeline,
    records: Any,
    n: int = 10,
    id_column: str = "id",
) -> dict[str, Any]:
    """Benchmark a full record-linkage pipeline.

    Parameters
    ----------
    pipeline : ReclinkPipeline
        A fully configured pipeline.
    records : DataFrame or list of dicts
        Dataset to deduplicate.
    n : int
        Number of iterations.
    id_column : str
        Column name for record identifiers.

    Returns
    -------
    dict
        Keys: ``total_ns``, ``per_run_ns``, ``runs_per_sec``,
        ``n_iterations``, ``n_records``, ``n_matches`` (from last run).
    """
    # Warm up
    last_results = pipeline.dedup(records, id_column=id_column)

    start = time.perf_counter_ns()
    for _ in range(n):
        last_results = pipeline.dedup(records, id_column=id_column)
    elapsed_ns = time.perf_counter_ns() - start

    per_run_ns = elapsed_ns / n if n > 0 else 0
    runs_per_sec = 1_000_000_000 / per_run_ns if per_run_ns > 0 else float("inf")

    n_records = len(records) if hasattr(records, "__len__") else 0
    n_matches = len(last_results) if hasattr(last_results, "__len__") else 0

    return {
        "total_ns": elapsed_ns,
        "per_run_ns": round(per_run_ns),
        "runs_per_sec": round(runs_per_sec, 2),
        "n_iterations": n,
        "n_records": n_records,
        "n_matches": n_matches,
    }


try:
    # Try to import hamming_similarity — it may fail on equal-length requirement
    _DEFAULT_METRICS_WITH_HAMMING = {**_DEFAULT_METRICS, "hamming": hamming_similarity}
except Exception:
    _DEFAULT_METRICS_WITH_HAMMING = _DEFAULT_METRICS
