"""Tests for Polars accessor integration."""

import pytest

pl = pytest.importorskip("polars")

import reclink  # noqa: E402


class TestSeriesAccessor:
    def test_match_best(self) -> None:
        s = pl.Series("names", ["John", "Jane", "Bob"])
        candidates = ["Jon", "Janet", "Robert"]
        result = s.reclink.match_best(candidates)
        assert len(result) == 3
        assert result[0] == "Jon"

    def test_phonetic(self) -> None:
        s = pl.Series("names", ["Smith", "Johnson"])
        result = s.reclink.phonetic("soundex")
        assert len(result) == 2
        assert result[0] == reclink.soundex("Smith")
