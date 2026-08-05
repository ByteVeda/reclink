"""Tests for async API wrappers (T5-3).

Uses asyncio.run() directly — no pytest-asyncio dependency needed.
"""

from __future__ import annotations

import asyncio

import numpy as np

import reclink.async_api as async_reclink
from reclink.async_api import AsyncPipeline
from reclink.pipeline import ReclinkPipeline


class TestAsyncCdist:
    def test_basic(self) -> None:
        result = asyncio.run(async_reclink.cdist(["hello", "world"], ["hello", "hi"]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert result[0, 0] > 0.9  # hello vs hello


class TestAsyncMatchBest:
    def test_basic(self) -> None:
        result = asyncio.run(async_reclink.match_best("hello", ["world", "hallo", "xyz"]))
        assert result is not None
        assert result[0] == "hallo"
        assert result[1] > 0.5

    def test_no_match(self) -> None:
        result = asyncio.run(async_reclink.match_best("hello", ["xyz"], threshold=0.99))
        assert result is None


class TestAsyncMatchBatch:
    def test_basic(self) -> None:
        results = asyncio.run(async_reclink.match_batch("hello", ["hallo", "world", "help"]))
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)


class TestAsyncPairwiseSimilarity:
    def test_basic(self) -> None:
        result = asyncio.run(
            async_reclink.pairwise_similarity(["hello", "world"], ["hello", "world"])
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] > 0.9  # hello vs hello


class TestAsyncPreprocessBatch:
    def test_basic(self) -> None:
        result = asyncio.run(
            async_reclink.preprocess_batch(["Hello World", "FOO BAR"], ["fold_case"])
        )
        assert result == ["hello world", "foo bar"]


class TestAsyncPipeline:
    def test_dedup(self) -> None:
        pipeline = (
            ReclinkPipeline.builder()
            .block_sorted_neighborhood("first_name", window=5)
            .compare_string("first_name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        records = [
            {"id": "1", "first_name": "John"},
            {"id": "2", "first_name": "Jon"},
        ]
        async_pipeline = AsyncPipeline(pipeline)
        results = asyncio.run(async_pipeline.dedup(records))
        assert isinstance(results, list)
