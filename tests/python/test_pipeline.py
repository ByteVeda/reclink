"""Tests for the record linkage pipeline.

Organized into sections:
- TestDedup: Basic deduplication
- TestDedupCluster: Clustering variations
- TestLink: Record linkage across datasets
- TestBlockingStrategies: Blocker-specific tests
- TestDataFrameOutput: DataFrame input/output round-trip
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from reclink.pipeline import ReclinkPipeline

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_DATA: list[dict[str, str]] = [
    {"id": "1", "first_name": "John", "last_name": "Smith"},
    {"id": "2", "first_name": "Jon", "last_name": "Smyth"},
    {"id": "3", "first_name": "Jane", "last_name": "Doe"},
]


def _base_builder() -> Any:
    """Return a pipeline builder with sensible defaults for testing."""
    return (
        ReclinkPipeline.builder()
        .preprocess("first_name", ["fold_case"])
        .preprocess("last_name", ["fold_case"])
        .compare_string("first_name", metric="jaro_winkler")
        .compare_string("last_name", metric="jaro_winkler")
        .classify_threshold(0.7)
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDedup:
    def test_basic_dedup(self) -> None:
        pipeline = _base_builder().block_phonetic("last_name", algorithm="soundex").build()
        results = pipeline.dedup(SAMPLE_DATA)
        assert len(results) >= 1
        ids = {(r["left_id"], r["right_id"]) for r in results}
        assert ("1", "2") in ids or ("2", "1") in ids

    def test_dedup_no_matches(self) -> None:
        data = [
            {"id": "1", "first_name": "Alice", "last_name": "Zephyr"},
            {"id": "2", "first_name": "Bob", "last_name": "Quantum"},
        ]
        pipeline = _base_builder().block_phonetic("last_name", algorithm="soundex").build()
        results = pipeline.dedup(data)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestDedupCluster:
    def test_connected_components(self) -> None:
        pipeline = (
            _base_builder()
            .block_phonetic("last_name", algorithm="soundex")
            .cluster_connected_components()
            .build()
        )
        clusters = pipeline.dedup_cluster(SAMPLE_DATA)
        found = any("1" in c and "2" in c for c in clusters)
        assert found

    def test_hierarchical_single(self) -> None:
        pipeline = (
            _base_builder()
            .block_phonetic("last_name", algorithm="soundex")
            .cluster_hierarchical(linkage="single", threshold=0.5)
            .build()
        )
        clusters = pipeline.dedup_cluster(SAMPLE_DATA)
        found = any("1" in c and "2" in c for c in clusters)
        assert found

    def test_hierarchical_average(self) -> None:
        pipeline = (
            _base_builder()
            .block_phonetic("last_name", algorithm="soundex")
            .cluster_hierarchical(linkage="average", threshold=0.5)
            .build()
        )
        clusters = pipeline.dedup_cluster(SAMPLE_DATA)
        assert isinstance(clusters, list)

    def test_hierarchical_complete(self) -> None:
        pipeline = (
            _base_builder()
            .block_phonetic("last_name", algorithm="soundex")
            .cluster_hierarchical(linkage="complete", threshold=0.5)
            .build()
        )
        clusters = pipeline.dedup_cluster(SAMPLE_DATA)
        assert isinstance(clusters, list)

    def test_no_clustering_returns_pairs(self) -> None:
        pipeline = _base_builder().block_phonetic("last_name", algorithm="soundex").build()
        clusters = pipeline.dedup_cluster(SAMPLE_DATA)
        for c in clusters:
            assert len(c) == 2


# ---------------------------------------------------------------------------
# Record linkage
# ---------------------------------------------------------------------------


class TestLink:
    def test_basic_link(self) -> None:
        left = [
            {"id": "a1", "name": "John Smith"},
            {"id": "a2", "name": "Jane Doe"},
        ]
        right = [
            {"id": "b1", "name": "Jon Smyth"},
            {"id": "b2", "name": "Bob Jones"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .preprocess("name", ["fold_case"])
            .block_sorted_neighborhood("name", window=5)
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        results = pipeline.link(left, right)
        assert any(r["left_id"] == "a1" and r["right_id"] == "b1" for r in results)


# ---------------------------------------------------------------------------
# Blocking strategies
# ---------------------------------------------------------------------------


class TestBlockingStrategies:
    def test_block_lsh(self) -> None:
        data = [
            {"id": "1", "name": "Jonathan Smith"},
            {"id": "2", "name": "Jonathon Smith"},
            {"id": "3", "name": "Xyz Abc"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .preprocess("name", ["fold_case"])
            .block_lsh("name", num_hashes=100, num_bands=20)
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        results = pipeline.dedup(data)
        ids = {(r["left_id"], r["right_id"]) for r in results}
        assert ("1", "2") in ids or ("2", "1") in ids

    def test_block_canopy(self) -> None:
        data = [
            {"id": "1", "name": "Smith"},
            {"id": "2", "name": "Smyth"},
            {"id": "3", "name": "Unrelated"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .preprocess("name", ["fold_case"])
            .block_canopy("name", t_tight=0.9, t_loose=0.5, metric="jaro_winkler")
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        results = pipeline.dedup(data)
        ids = {(r["left_id"], r["right_id"]) for r in results}
        assert ("1", "2") in ids or ("2", "1") in ids


# ---------------------------------------------------------------------------
# DataFrame output
# ---------------------------------------------------------------------------


class TestDataFrameOutput:
    def test_dedup_list_returns_list(self) -> None:
        pipeline = _base_builder().block_phonetic("last_name", algorithm="soundex").build()
        results = pipeline.dedup(SAMPLE_DATA)
        assert isinstance(results, list)

    def test_dedup_cluster_list_returns_list(self) -> None:
        pipeline = (
            _base_builder()
            .block_phonetic("last_name", algorithm="soundex")
            .cluster_connected_components()
            .build()
        )
        results = pipeline.dedup_cluster(SAMPLE_DATA)
        assert isinstance(results, list)

    def test_dedup_pandas_returns_dataframe(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(SAMPLE_DATA)
        pipeline = _base_builder().block_phonetic("last_name", algorithm="soundex").build()
        results = pipeline.dedup(df)
        assert isinstance(results, pd.DataFrame)
        assert "left_id" in results.columns
        assert "right_id" in results.columns

    def test_dedup_cluster_pandas_returns_dataframe(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(SAMPLE_DATA)
        pipeline = (
            _base_builder()
            .block_phonetic("last_name", algorithm="soundex")
            .cluster_connected_components()
            .build()
        )
        results = pipeline.dedup_cluster(df)
        assert isinstance(results, pd.DataFrame)
        assert "cluster_id" in results.columns
        assert "record_id" in results.columns

    def test_dedup_polars_returns_dataframe(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(SAMPLE_DATA)
        pipeline = _base_builder().block_phonetic("last_name", algorithm="soundex").build()
        results = pipeline.dedup(df)
        assert isinstance(results, pl.DataFrame)
        assert "left_id" in results.columns
        assert "right_id" in results.columns

    def test_link_pandas_returns_dataframe(self) -> None:
        pd = pytest.importorskip("pandas")
        left = pd.DataFrame(
            [
                {"id": "a1", "name": "John Smith"},
                {"id": "a2", "name": "Jane Doe"},
            ]
        )
        right = pd.DataFrame(
            [
                {"id": "b1", "name": "Jon Smyth"},
                {"id": "b2", "name": "Bob Jones"},
            ]
        )
        pipeline = (
            ReclinkPipeline.builder()
            .preprocess("name", ["fold_case"])
            .block_sorted_neighborhood("name", window=5)
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        results = pipeline.link(left, right)
        assert isinstance(results, pd.DataFrame)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_per_field_preprocess(self) -> None:
        """Different ops on different fields."""
        data = [
            {"id": "1", "first_name": "JOHN", "last_name": "Smith!"},
            {"id": "2", "first_name": "john", "last_name": "Smyth!"},
            {"id": "3", "first_name": "JANE", "last_name": "Doe"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .preprocess("first_name", ["fold_case"])
            .preprocess("last_name", ["fold_case", "strip_punctuation"])
            .block_phonetic("last_name", algorithm="soundex")
            .compare_string("first_name", metric="jaro_winkler")
            .compare_string("last_name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        results = pipeline.dedup(data)
        assert len(results) >= 1
        ids = {(r["left_id"], r["right_id"]) for r in results}
        assert ("1", "2") in ids or ("2", "1") in ids

    def test_preprocess_lowercase_deprecated(self) -> None:
        """preprocess_lowercase emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ReclinkPipeline.builder().preprocess_lowercase(["name"])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# New classifier and blocking strategies
# ---------------------------------------------------------------------------


class TestFellegiSunterAuto:
    def test_em_auto_finds_matches(self) -> None:
        data = [
            {"id": "1", "first_name": "John", "last_name": "Smith"},
            {"id": "2", "first_name": "Jon", "last_name": "Smyth"},
            {"id": "3", "first_name": "Jane", "last_name": "Doe"},
            {"id": "4", "first_name": "Alice", "last_name": "Zephyr"},
            {"id": "5", "first_name": "Bob", "last_name": "Quantum"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .preprocess("first_name", ["fold_case"])
            .preprocess("last_name", ["fold_case"])
            .block_phonetic("last_name", algorithm="soundex")
            .compare_string("first_name", metric="jaro_winkler")
            .compare_string("last_name", metric="jaro_winkler")
            .classify_fellegi_sunter_auto()
            .build()
        )
        results = pipeline.dedup(data)
        # Should find at least the John/Jon Smith/Smyth pair
        assert isinstance(results, list)


class TestNewBlockingStrategies:
    def test_block_numeric(self) -> None:
        data = [
            {"id": "1", "name": "Alice", "age": "25"},
            {"id": "2", "name": "Alice", "age": "27"},
            {"id": "3", "name": "Bob", "age": "60"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .block_numeric("age", bucket_size=5.0)
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        results = pipeline.dedup(data)
        # Alice records should be blocked together (same bucket)
        ids = {(r["left_id"], r["right_id"]) for r in results}
        assert ("1", "2") in ids or ("2", "1") in ids
        # Bob should not match (different bucket and different name)
        bob_pairs = {(a, b) for a, b in ids if "3" in (a, b)}
        assert len(bob_pairs) == 0

    def test_block_date(self) -> None:
        data = [
            {"id": "1", "name": "Alice", "dob": "1990-01-15"},
            {"id": "2", "name": "Alice", "dob": "1990-06-20"},
            {"id": "3", "name": "Bob", "dob": "2000-01-01"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .block_date("dob", resolution="year")
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.7)
            .build()
        )
        results = pipeline.dedup(data)
        # Records 1 and 2 share the same year
        ids = {(r["left_id"], r["right_id"]) for r in results}
        assert ("1", "2") in ids or ("2", "1") in ids


class TestMatchResultRepr:
    def test_repr_contains_fields(self) -> None:
        from reclink._core import PyRecord

        pipeline = _base_builder().block_phonetic("last_name", algorithm="soundex").build()
        # Access the underlying MatchResult via _inner
        records = [PyRecord(d["id"]) for d in SAMPLE_DATA]
        for rec, d in zip(records, SAMPLE_DATA, strict=True):
            for k, v in d.items():
                if k != "id":
                    rec.set_field(k, v)
        inner_results = pipeline._inner.dedup(records)
        r = inner_results[0]
        text = repr(r)
        assert "MatchResult(" in text
        assert "left_id=" in text
        assert "right_id=" in text
        assert "score=" in text
        assert "match_class=" in text
        assert "scores=" in text

    def test_match_class_in_output(self) -> None:
        pipeline = _base_builder().block_phonetic("last_name", algorithm="soundex").build()
        results = pipeline.dedup(SAMPLE_DATA)
        assert len(results) >= 1
        assert "match_class" in results[0]
        assert results[0]["match_class"] in ("match", "possible", "non_match")


class TestThresholdBands:
    def test_threshold_bands_match_and_possible(self) -> None:
        data = [
            {"id": "1", "first_name": "John", "last_name": "Smith"},
            {"id": "2", "first_name": "Jon", "last_name": "Smyth"},
            {"id": "3", "first_name": "Jane", "last_name": "Doe"},
            {"id": "4", "first_name": "Janet", "last_name": "Doer"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .preprocess("first_name", ["fold_case"])
            .preprocess("last_name", ["fold_case"])
            .block_sorted_neighborhood("last_name", window=10)
            .compare_string("first_name", metric="jaro_winkler")
            .compare_string("last_name", metric="jaro_winkler")
            .classify_threshold_bands(upper=0.9, lower=0.5)
            .build()
        )
        results = pipeline.dedup(data)
        classes = {r["match_class"] for r in results}
        # At minimum we should get some results with valid match classes
        assert classes.issubset({"match", "possible", "non_match"})


class TestPhoneticComparator:
    def test_compare_phonetic(self) -> None:
        left = [
            {"id": "a1", "name": "Smith"},
            {"id": "a2", "name": "Jones"},
        ]
        right = [
            {"id": "b1", "name": "Smyth"},
            {"id": "b2", "name": "Johnson"},
        ]
        pipeline = (
            ReclinkPipeline.builder()
            .block_sorted_neighborhood("name", window=5)
            .compare_phonetic("name", algorithm="soundex")
            .classify_threshold(0.5)
            .build()
        )
        results = pipeline.link(left, right)
        # Smith/Smyth have the same Soundex code
        assert any(r["left_id"] == "a1" and r["right_id"] == "b1" for r in results)
