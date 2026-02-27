"""Tests for custom metric registration (T5-1)."""

from __future__ import annotations

import pytest

import reclink


def _exact_match(a: str, b: str) -> float:
    """Simple test metric: 1.0 if equal, 0.0 otherwise."""
    return 1.0 if a == b else 0.0


class TestRegisterAndUse:
    def setup_method(self) -> None:
        # Clean up any leftover registrations
        for name in reclink.list_custom_metrics():
            reclink.unregister_metric(name)

    def teardown_method(self) -> None:
        for name in reclink.list_custom_metrics():
            reclink.unregister_metric(name)

    def test_register_and_use_cdist(self) -> None:
        reclink.register_metric("exact_eq", _exact_match)
        result = reclink.cdist(["hello", "world"], ["hello", "xyz"], scorer="exact_eq")
        assert result[0, 0] == 1.0  # hello == hello
        assert result[0, 1] == 0.0  # hello != xyz
        assert result[1, 0] == 0.0  # world != hello

    def test_register_and_use_match_best(self) -> None:
        reclink.register_metric("exact_eq2", _exact_match)
        result = reclink.match_best("hello", ["world", "hello", "hi"], scorer="exact_eq2")
        assert result is not None
        assert result[0] == "hello"
        assert result[1] == 1.0

    def test_unregister(self) -> None:
        reclink.register_metric("temp_metric", _exact_match)
        assert reclink.unregister_metric("temp_metric")
        # Should no longer be usable
        with pytest.raises(ValueError, match="unknown metric"):
            reclink.cdist(["a"], ["b"], scorer="temp_metric")


class TestDecoratorSyntax:
    def setup_method(self) -> None:
        for name in reclink.list_custom_metrics():
            reclink.unregister_metric(name)

    def teardown_method(self) -> None:
        for name in reclink.list_custom_metrics():
            reclink.unregister_metric(name)

    def test_decorator_with_name(self) -> None:
        @reclink.register_metric("my_metric")
        def _my_func(a: str, b: str) -> float:
            return 0.5

        assert "my_metric" in reclink.list_custom_metrics()
        result = reclink.match_best("a", ["b"], scorer="my_metric")
        assert result is not None
        assert result[1] == 0.5

    def test_bare_decorator(self) -> None:
        @reclink.register_metric
        def length_ratio(a: str, b: str) -> float:
            la, lb = len(a), len(b)
            if la == 0 and lb == 0:
                return 1.0
            return min(la, lb) / max(la, lb)

        assert "length_ratio" in reclink.list_custom_metrics()

    def test_direct_call(self) -> None:
        reclink.register_metric("direct_metric", _exact_match)
        assert "direct_metric" in reclink.list_custom_metrics()


class TestListCustomMetrics:
    def setup_method(self) -> None:
        for name in reclink.list_custom_metrics():
            reclink.unregister_metric(name)

    def teardown_method(self) -> None:
        for name in reclink.list_custom_metrics():
            reclink.unregister_metric(name)

    def test_initially_empty(self) -> None:
        assert reclink.list_custom_metrics() == []

    def test_registration_appears_in_list(self) -> None:
        reclink.register_metric("listed_metric", _exact_match)
        names = reclink.list_custom_metrics()
        assert "listed_metric" in names

    def test_unregister_removes_from_list(self) -> None:
        reclink.register_metric("gone_metric", _exact_match)
        reclink.unregister_metric("gone_metric")
        assert "gone_metric" not in reclink.list_custom_metrics()


class TestCannotOverrideBuiltin:
    def test_jaro_winkler(self) -> None:
        with pytest.raises(ValueError, match="built-in"):
            reclink.register_metric("jaro_winkler", _exact_match)

    def test_levenshtein(self) -> None:
        with pytest.raises(ValueError, match="built-in"):
            reclink.register_metric("levenshtein", _exact_match)


class TestUnregisterUnknown:
    def test_returns_false(self) -> None:
        assert reclink.unregister_metric("nonexistent") is False
