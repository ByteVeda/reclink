#!/usr/bin/env python3
"""Benchmark reclink against rapidfuzz, jellyfish, and thefuzz.

Usage::

    pip install rapidfuzz jellyfish thefuzz
    python benchmarks/compare.py

Generates a comparison table suitable for inclusion in README.md.
"""

from __future__ import annotations

import statistics
import sys
import timeit
from typing import Any

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

PAIRS = [
    ("kitten", "sitting"),
    ("Saturday", "Sunday"),
    ("Robert", "Rupert"),
    ("John Smith", "Jon Smyth"),
    ("University of California", "Univ. of Cal."),
    ("123 Main Street", "123 Main St"),
    ("Python programming language", "Python coding language"),
    ("New York City", "New York"),
    ("", "hello"),
    ("identical", "identical"),
]

BATCH_QUERY = "John Smith"
BATCH_CANDIDATES = [
    "Jon Smith", "Jane Doe", "John Smyth", "Robert Johnson", "Johnny Smith",
    "Smith, John", "J. Smith", "Jonathan Smith", "John S.", "James Smith",
    "Joan Smith", "John Schmidt", "Johnson Smith", "Johan Smit", "John Smithson",
    "Jean Smith", "Juan Smith", "John Smiley", "John Smooth", "John Smart",
] * 50  # 1000 candidates

NUMBER = 500  # iterations for pairwise benchmarks
BATCH_NUMBER = 50  # iterations for batch benchmarks


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_pairwise(
    func: Any, pairs: list[tuple[str, str]], number: int = NUMBER
) -> float:
    """Return median microseconds per pair for a pairwise function."""
    times = timeit.repeat(
        lambda: [func(a, b) for a, b in pairs],
        number=number,
        repeat=5,
    )
    median = statistics.median(times)
    us_per_pair = (median / number / len(pairs)) * 1_000_000
    return us_per_pair


def bench_batch(func: Any, number: int = BATCH_NUMBER) -> float:
    """Return median microseconds per candidate for a batch function."""
    n_cand = len(BATCH_CANDIDATES)
    times = timeit.repeat(
        lambda: func(BATCH_QUERY, BATCH_CANDIDATES),
        number=number,
        repeat=5,
    )
    median = statistics.median(times)
    us_per_candidate = (median / number / n_cand) * 1_000_000
    return us_per_candidate


# ---------------------------------------------------------------------------
# Library wrappers
# ---------------------------------------------------------------------------

def get_reclink_funcs() -> dict[str, Any]:
    """Get reclink benchmark functions."""
    import reclink
    return {
        "levenshtein": lambda a, b: reclink.levenshtein(a, b),
        "jaro": lambda a, b: reclink.jaro(a, b),
        "jaro_winkler": lambda a, b: reclink.jaro_winkler(a, b),
        "damerau_levenshtein": lambda a, b: reclink.damerau_levenshtein(a, b),
        "hamming": None,  # requires equal length
        "batch_jw": lambda q, c: reclink.match_batch(q, c, scorer="jaro_winkler"),
    }


def get_rapidfuzz_funcs() -> dict[str, Any] | None:
    """Get rapidfuzz benchmark functions."""
    try:
        from rapidfuzz import distance, fuzz, process
        return {
            "levenshtein": lambda a, b: distance.Levenshtein.distance(a, b),
            "jaro": lambda a, b: distance.Jaro.similarity(a, b),
            "jaro_winkler": lambda a, b: distance.JaroWinkler.similarity(a, b),
            "damerau_levenshtein": lambda a, b: distance.DamerauLevenshtein.distance(a, b),
            "hamming": None,
            "batch_jw": lambda q, c: process.extract(
                q, c, scorer=distance.JaroWinkler.similarity, limit=len(c),
            ),
        }
    except ImportError:
        return None


def get_jellyfish_funcs() -> dict[str, Any] | None:
    """Get jellyfish benchmark functions."""
    try:
        import jellyfish
        return {
            "levenshtein": lambda a, b: jellyfish.levenshtein_distance(a, b),
            "jaro": lambda a, b: jellyfish.jaro_similarity(a, b),
            "jaro_winkler": lambda a, b: jellyfish.jaro_winkler_similarity(a, b),
            "damerau_levenshtein": lambda a, b: jellyfish.damerau_levenshtein_distance(a, b),
            "hamming": None,
            "batch_jw": None,  # jellyfish has no batch API
        }
    except ImportError:
        return None


def get_thefuzz_funcs() -> dict[str, Any] | None:
    """Get thefuzz benchmark functions."""
    try:
        from thefuzz import fuzz, process
        return {
            "levenshtein": None,  # thefuzz doesn't expose raw levenshtein
            "jaro": None,
            "jaro_winkler": None,
            "damerau_levenshtein": None,
            "hamming": None,
            "batch_jw": lambda q, c: process.extract(q, c, scorer=fuzz.WRatio, limit=len(c)),
        }
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_cell(us: float | None) -> str:
    """Format a cell value."""
    if us is None:
        return "—"
    return f"{us:.2f}"


def speedup(reclink_us: float, other_us: float | None) -> str:
    """Format speedup ratio."""
    if other_us is None or other_us == 0:
        return "—"
    ratio = other_us / reclink_us
    if ratio >= 1:
        return f"**{ratio:.1f}x faster**"
    return f"{1/ratio:.1f}x slower"


def main() -> None:
    """Run benchmarks and print results."""
    print("Benchmarking reclink vs other libraries...\n")

    libs: dict[str, dict[str, Any] | None] = {
        "reclink": get_reclink_funcs(),
        "rapidfuzz": get_rapidfuzz_funcs(),
        "jellyfish": get_jellyfish_funcs(),
        "thefuzz": get_thefuzz_funcs(),
    }

    available = {k: v for k, v in libs.items() if v is not None}
    if len(available) < 2:
        missing = [k for k, v in libs.items() if v is None and k != "reclink"]
        print(f"Install comparison libraries: pip install {' '.join(missing)}")
        print("Running reclink-only benchmarks...\n")

    metrics = ["levenshtein", "jaro", "jaro_winkler", "damerau_levenshtein"]

    # Pairwise benchmarks
    print(f"### Pairwise ({len(PAIRS)} pairs, {NUMBER} iterations)\n")
    header = "| Metric |"
    sep = "|--------|"
    for lib_name in available:
        header += f" {lib_name} (us/pair) |"
        sep += "---:|"
    if "reclink" in available and len(available) > 1:
        for lib_name in available:
            if lib_name != "reclink":
                header += f" vs {lib_name} |"
                sep += "---:|"
    print(header)
    print(sep)

    pairwise_results: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        row = f"| {metric} |"
        pairwise_results[metric] = {}
        for lib_name, funcs in available.items():
            func = funcs.get(metric)
            if func is not None:
                us = bench_pairwise(func, PAIRS)
                pairwise_results[metric][lib_name] = us
                row += f" {format_cell(us)} |"
            else:
                pairwise_results[metric][lib_name] = None
                row += " — |"
        if "reclink" in available and len(available) > 1:
            reclink_us = pairwise_results[metric].get("reclink")
            for lib_name in available:
                if lib_name != "reclink" and reclink_us is not None:
                    row += f" {speedup(reclink_us, pairwise_results[metric].get(lib_name))} |"
        print(row)

    # Batch benchmarks
    print(f"\n### Batch match ({len(BATCH_CANDIDATES)} candidates, {BATCH_NUMBER} iterations)\n")
    header = "| Operation |"
    sep = "|-----------|"
    for lib_name in available:
        header += f" {lib_name} (us/cand) |"
        sep += "---:|"
    if "reclink" in available and len(available) > 1:
        for lib_name in available:
            if lib_name != "reclink":
                header += f" vs {lib_name} |"
                sep += "---:|"
    print(header)
    print(sep)

    row = "| match_batch (jaro_winkler) |"
    batch_results: dict[str, float | None] = {}
    for lib_name, funcs in available.items():
        func = funcs.get("batch_jw") if funcs else None
        if func is not None:
            us = bench_batch(func)
            batch_results[lib_name] = us
            row += f" {format_cell(us)} |"
        else:
            batch_results[lib_name] = None
            row += " — |"
    if "reclink" in available and len(available) > 1:
        reclink_us = batch_results.get("reclink")
        for lib_name in available:
            if lib_name != "reclink" and reclink_us is not None:
                row += f" {speedup(reclink_us, batch_results.get(lib_name))} |"
    print(row)

    print("\n_Times in microseconds per operation. Lower is better._")
    print(f"_Benchmark ran on {len(PAIRS)} string pairs and {len(BATCH_CANDIDATES)} candidates._")


if __name__ == "__main__":
    main()
