"""Async wrappers for CPU-bound reclink operations.

All functions use ``asyncio.to_thread()`` to offload work to a thread pool,
releasing the event loop while Rust + Rayon compute in the background.

Usage
-----
>>> import asyncio
>>> import reclink.async_api as async_reclink
>>> result = asyncio.run(async_reclink.cdist(["a"], ["b"], scorer="jaro_winkler"))
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import reclink._core as _core
from reclink.streaming import match_stream_async as match_stream_async

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from reclink._core import Scorer


# ---------------------------------------------------------------------------
# Metric operations
# ---------------------------------------------------------------------------


async def cdist(
    a: Sequence[str],
    b: Sequence[str],
    scorer: Scorer = "jaro_winkler",
    workers: int | None = None,
) -> NDArray[np.float64]:
    """Async version of :func:`reclink.cdist`."""
    return await asyncio.to_thread(_core.cdist, list(a), list(b), scorer, workers)


async def match_best(
    query: str,
    candidates: list[str],
    scorer: Scorer = "jaro_winkler",
    threshold: float | None = None,
    workers: int | None = None,
) -> tuple[str, float, int] | None:
    """Async version of :func:`reclink.match_best`."""
    return await asyncio.to_thread(_core.match_best, query, candidates, scorer, threshold, workers)


async def match_batch(
    query: str,
    candidates: list[str],
    scorer: Scorer = "jaro_winkler",
    threshold: float | None = None,
    limit: int | None = None,
    workers: int | None = None,
) -> list[tuple[str, float, int]]:
    """Async version of :func:`reclink.match_batch`."""
    return await asyncio.to_thread(
        _core.match_batch, query, candidates, scorer, threshold, limit, workers
    )


async def match_best_arrow(
    query: str,
    candidates: list[str],
    scorer: Scorer = "jaro_winkler",
    threshold: float = 0.0,
) -> tuple[str, float, int] | None:
    """Async version of :func:`reclink.match_best_arrow`."""
    return await asyncio.to_thread(_core.match_best_arrow, query, candidates, scorer, threshold)


async def match_batch_arrow(
    query: str,
    candidates: list[str],
    scorer: Scorer = "jaro_winkler",
    threshold: float = 0.0,
    limit: int | None = None,
) -> list[tuple[str, float, int]]:
    """Async version of :func:`reclink.match_batch_arrow`."""
    return await asyncio.to_thread(
        _core.match_batch_arrow, query, candidates, scorer, threshold, limit
    )


async def pairwise_similarity(
    a: list[str],
    b: list[str],
    scorer: Scorer = "jaro_winkler",
) -> list[float]:
    """Async version of :func:`reclink.pairwise_similarity`."""
    return await asyncio.to_thread(_core.pairwise_similarity, a, b, scorer)


async def preprocess_batch(
    strings: list[str],
    operations: list[str],
) -> list[str]:
    """Async version of :func:`reclink.preprocess_batch`."""
    return await asyncio.to_thread(_core.preprocess_batch, strings, operations)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AsyncPipeline:
    """Async wrapper around :class:`reclink.pipeline.ReclinkPipeline`.

    Parameters
    ----------
    pipeline : ReclinkPipeline
        A fully-configured synchronous pipeline.
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    async def dedup(
        self,
        data: Any,
        id_column: str = "id",
    ) -> Any:
        """Async version of ``pipeline.dedup()``."""
        return await asyncio.to_thread(self._pipeline.dedup, data, id_column)

    async def dedup_cluster(
        self,
        data: Any,
        id_column: str = "id",
    ) -> Any:
        """Async version of ``pipeline.dedup_cluster()``."""
        return await asyncio.to_thread(self._pipeline.dedup_cluster, data, id_column)

    async def link(
        self,
        left: Any,
        right: Any,
        id_column: str = "id",
    ) -> Any:
        """Async version of ``pipeline.link()``."""
        return await asyncio.to_thread(self._pipeline.link, left, right, id_column)
