"""Tests for custom plugin registration: blockers, comparators, classifiers, preprocessors."""

from __future__ import annotations

import pytest

import reclink

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleanup_plugins() -> None:
    """Remove all registered custom plugins."""
    for name in reclink.list_custom_blockers():
        reclink.unregister_blocker(name)
    for name in reclink.list_custom_comparators():
        reclink.unregister_comparator(name)
    for name in reclink.list_custom_classifiers():
        reclink.unregister_classifier(name)
    for name in reclink.list_custom_preprocessors():
        reclink.unregister_preprocessor(name)
    for name in reclink.list_custom_metrics():
        reclink.unregister_metric(name)


class _AllPairsBlocker:
    """Blocker that returns all pairs — useful for testing."""

    def block_dedup(self, records: list[dict[str, str]]) -> list[tuple[int, int]]:
        pairs = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                pairs.append((i, j))
        return pairs

    def block_link(
        self, left: list[dict[str, str]], right: list[dict[str, str]]
    ) -> list[tuple[int, int]]:
        pairs = []
        for i in range(len(left)):
            for j in range(len(right)):
                pairs.append((i, j))
        return pairs


# ---------------------------------------------------------------------------
# Custom Blockers
# ---------------------------------------------------------------------------


class TestCustomBlocker:
    def setup_method(self) -> None:
        _cleanup_plugins()

    def teardown_method(self) -> None:
        _cleanup_plugins()

    def test_register_and_list(self) -> None:
        reclink.register_blocker("all_pairs", _AllPairsBlocker())
        assert "all_pairs" in reclink.list_custom_blockers()

    def test_unregister(self) -> None:
        reclink.register_blocker("temp_blocker", _AllPairsBlocker())
        assert reclink.unregister_blocker("temp_blocker")
        assert "temp_blocker" not in reclink.list_custom_blockers()
        assert not reclink.unregister_blocker("temp_blocker")

    def test_blocker_in_pipeline_dedup(self) -> None:
        reclink.register_blocker("all_pairs", _AllPairsBlocker())

        pipeline = (
            reclink.pipeline.PipelineBuilder()
            .block_custom("all_pairs")
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.5)
            .build()
        )
        data = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Alicia"},
            {"id": "3", "name": "Bob"},
        ]
        results = pipeline.dedup(data)
        # All 3 pairs should be compared; Alice/Alicia should match
        assert any(
            (r["left_id"] == "1" and r["right_id"] == "2")
            or (r["left_id"] == "2" and r["right_id"] == "1")
            for r in results
        )

    def test_blocker_in_pipeline_link(self) -> None:
        reclink.register_blocker("all_pairs", _AllPairsBlocker())

        pipeline = (
            reclink.pipeline.PipelineBuilder()
            .block_custom("all_pairs")
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.5)
            .build()
        )
        left = [{"id": "1", "name": "Alice"}]
        right = [{"id": "2", "name": "Alicia"}]
        results = pipeline.link(left, right)
        assert len(results) > 0

    def test_unregistered_blocker_fails(self) -> None:
        pipeline = (
            reclink.pipeline.PipelineBuilder()
            .block_custom("nonexistent")
            .compare_string("name")
            .classify_threshold(0.5)
            .build()
        )
        with pytest.raises(ValueError, match="unknown custom blocker"):
            pipeline.dedup([{"id": "1", "name": "Alice"}])


# ---------------------------------------------------------------------------
# Custom Comparators
# ---------------------------------------------------------------------------


class TestCustomComparator:
    def setup_method(self) -> None:
        _cleanup_plugins()

    def teardown_method(self) -> None:
        _cleanup_plugins()

    def test_register_and_list(self) -> None:
        reclink.register_comparator("exact_cmp", lambda a, b: 1.0 if a == b else 0.0)
        assert "exact_cmp" in reclink.list_custom_comparators()

    def test_unregister(self) -> None:
        reclink.register_comparator("temp_cmp", lambda a, b: 0.5)
        assert reclink.unregister_comparator("temp_cmp")
        assert not reclink.unregister_comparator("temp_cmp")

    def test_comparator_in_pipeline(self) -> None:
        reclink.register_comparator("exact_cmp", lambda a, b: 1.0 if a == b else 0.0)

        pipeline = (
            reclink.pipeline.PipelineBuilder()
            .block_exact("city")
            .compare_custom("name", "exact_cmp")
            .classify_threshold(0.5)
            .build()
        )
        data = [
            {"id": "1", "name": "Alice", "city": "NYC"},
            {"id": "2", "name": "Alice", "city": "NYC"},
            {"id": "3", "name": "Bob", "city": "NYC"},
        ]
        results = pipeline.dedup(data)
        # Alice/Alice should be exact match (score=1.0), Bob/Alice score=0.0
        match_pairs = {(r["left_id"], r["right_id"]) for r in results}
        assert ("1", "2") in match_pairs or ("2", "1") in match_pairs

    def test_invalid_callable_rejected(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            reclink.register_comparator("bad_cmp", lambda a, b: "not_a_float")


# ---------------------------------------------------------------------------
# Custom Classifiers
# ---------------------------------------------------------------------------


class TestCustomClassifier:
    def setup_method(self) -> None:
        _cleanup_plugins()

    def teardown_method(self) -> None:
        _cleanup_plugins()

    def test_register_and_list(self) -> None:
        def my_classifier(scores: list[float]) -> tuple[float, str]:
            avg = sum(scores) / len(scores) if scores else 0.0
            return (avg, "match" if avg >= 0.9 else "non_match")

        reclink.register_classifier("strict", my_classifier)
        assert "strict" in reclink.list_custom_classifiers()

    def test_unregister(self) -> None:
        reclink.register_classifier("temp_cls", lambda s: (sum(s) / len(s), "match"))
        assert reclink.unregister_classifier("temp_cls")
        assert not reclink.unregister_classifier("temp_cls")

    def test_classifier_in_pipeline(self) -> None:
        def always_match(scores: list[float]) -> tuple[float, str]:
            return (1.0, "match")

        reclink.register_classifier("always_match", always_match)

        pipeline = (
            reclink.pipeline.PipelineBuilder()
            .block_exact("city")
            .compare_string("name", metric="jaro_winkler")
            .classify_custom("always_match")
            .build()
        )
        data = [
            {"id": "1", "name": "Alice", "city": "NYC"},
            {"id": "2", "name": "Bob", "city": "NYC"},
        ]
        results = pipeline.dedup(data)
        # Custom classifier always returns match, so even dissimilar names match
        assert len(results) == 1
        assert results[0]["match_class"] == "match"

    def test_three_class_classifier(self) -> None:
        def banded(scores: list[float]) -> tuple[float, str]:
            avg = sum(scores) / len(scores) if scores else 0.0
            if avg >= 0.9:
                return (avg, "match")
            if avg >= 0.5:
                return (avg, "possible")
            return (avg, "non_match")

        reclink.register_classifier("banded", banded)

        pipeline = (
            reclink.pipeline.PipelineBuilder()
            .block_exact("city")
            .compare_string("name", metric="jaro_winkler")
            .classify_custom("banded")
            .build()
        )
        data = [
            {"id": "1", "name": "Alice", "city": "NYC"},
            {"id": "2", "name": "Alicia", "city": "NYC"},
        ]
        results = pipeline.dedup(data)
        assert len(results) >= 1
        assert results[0]["match_class"] in ("match", "possible")

    def test_invalid_classifier_rejected(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            reclink.register_classifier("bad_cls", lambda s: "just_a_string")


# ---------------------------------------------------------------------------
# Custom Preprocessors
# ---------------------------------------------------------------------------


class TestCustomPreprocessor:
    def setup_method(self) -> None:
        _cleanup_plugins()

    def teardown_method(self) -> None:
        _cleanup_plugins()

    def test_register_and_list(self) -> None:
        reclink.register_preprocessor("upper", lambda s: s.upper())
        assert "upper" in reclink.list_custom_preprocessors()

    def test_unregister(self) -> None:
        reclink.register_preprocessor("temp_prep", lambda s: s)
        assert reclink.unregister_preprocessor("temp_prep")
        assert not reclink.unregister_preprocessor("temp_prep")

    def test_preprocessor_in_pipeline(self) -> None:
        import re

        reclink.register_preprocessor("strip_numbers", lambda s: re.sub(r"\d+", "", s))

        pipeline = (
            reclink.pipeline.PipelineBuilder()
            .preprocess("name", ["fold_case", "custom:strip_numbers"])
            .block_exact("city")
            .compare_string("name", metric="jaro_winkler")
            .classify_threshold(0.8)
            .build()
        )
        data = [
            {"id": "1", "name": "Alice123", "city": "NYC"},
            {"id": "2", "name": "alice", "city": "NYC"},
        ]
        results = pipeline.dedup(data)
        # After stripping numbers and folding case, "Alice123" -> "alice" == "alice"
        assert len(results) == 1
        assert results[0]["score"] == 1.0

    def test_invalid_preprocessor_rejected(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            reclink.register_preprocessor("bad_prep", lambda s: 42)


# ---------------------------------------------------------------------------
# Combined plugin pipeline
# ---------------------------------------------------------------------------


class TestCombinedPlugins:
    def setup_method(self) -> None:
        _cleanup_plugins()

    def teardown_method(self) -> None:
        _cleanup_plugins()

    def test_all_custom_components(self) -> None:
        """Pipeline using custom blocker + comparator + classifier + preprocessor."""
        reclink.register_blocker("all_pairs", _AllPairsBlocker())
        reclink.register_comparator("length_cmp", lambda a, b: 1.0 if len(a) == len(b) else 0.0)
        reclink.register_preprocessor("trim_spaces", lambda s: s.strip())

        def lenient(scores: list[float]) -> tuple[float, str]:
            avg = sum(scores) / len(scores) if scores else 0.0
            return (avg, "match" if avg > 0.5 else "non_match")

        reclink.register_classifier("lenient", lenient)

        pipeline = (
            reclink.pipeline.PipelineBuilder()
            .preprocess("name", ["custom:trim_spaces"])
            .block_custom("all_pairs")
            .compare_custom("name", "length_cmp")
            .classify_custom("lenient")
            .build()
        )

        data = [
            {"id": "1", "name": "  Alice  "},
            {"id": "2", "name": "Alicia"},
            {"id": "3", "name": " Bob "},
        ]
        results = pipeline.dedup(data)
        # After trimming: "Alice" (5), "Alicia" (6), "Bob" (3)
        # length_cmp: Alice/Alicia=0, Alice/Bob=0, Alicia/Bob=0
        # All non_match since no equal lengths
        assert all(r["match_class"] == "non_match" for r in results)
