"""Streaming API for lazy/chunked candidate matching.

Processes candidates from iterators without materializing the full dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reclink._core import PyStreamingMatcher

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _enumerate_chunks(iterable: Iterable[str], chunk_size: int) -> Iterator[tuple[int, list[str]]]:
    """Yield (chunk_start_index, chunk) tuples from an iterable."""
    chunk: list[str] = []
    start = 0
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield start, chunk
            start += len(chunk)
            chunk = []
    if chunk:
        yield start, chunk


def match_stream(
    query: str,
    candidates: Iterable[str],
    scorer: str = "jaro_winkler",
    threshold: float | None = None,
    chunk_size: int = 1000,
) -> Iterator[tuple[str, float, int]]:
    """Lazily match a query against an iterable of candidates.

    Processes candidates in chunks for efficiency while maintaining
    lazy evaluation. Yields (matched_string, score, global_index) tuples.

    Parameters
    ----------
    query : str
        The string to match.
    candidates : Iterable[str]
        An iterable of candidate strings. Can be a generator.
    scorer : str, optional
        Metric name (default "jaro_winkler").
    threshold : float or None, optional
        Minimum similarity score to yield (default None, yields all).
    chunk_size : int, optional
        Number of candidates to process at a time (default 1000).

    Yields
    ------
    tuple of (str, float, int)
        Tuples of (matched_string, score, global_index).

    Examples
    --------
    >>> list(match_stream("hello", ["hello", "world", "help"], threshold=0.5))
    [('hello', 1.0, 0), ('help', ..., 2)]
    """
    matcher = PyStreamingMatcher(query, scorer, threshold)
    for chunk_start, chunk in _enumerate_chunks(candidates, chunk_size):
        for local_idx, score in matcher.score_chunk(chunk):
            yield (chunk[local_idx], score, chunk_start + local_idx)
