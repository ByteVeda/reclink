"""Tests for fuzzy_join API."""

import pandas as pd
import pytest

from reclink.fuzzy_join import fuzzy_join


class TestFuzzyJoinPandas:
    def test_basic_inner(self) -> None:
        left = pd.DataFrame({"name": ["Jon", "Jane"]})
        right = pd.DataFrame({"name": ["John", "Janet"]})
        result = fuzzy_join(left, right, on="name", threshold=0.7)
        assert len(result) > 0
        assert "_score" in result.columns
        assert "name_right" in result.columns

    def test_left_on_right_on(self) -> None:
        left = pd.DataFrame({"first": ["Jon"]})
        right = pd.DataFrame({"nombre": ["John"]})
        result = fuzzy_join(left, right, left_on="first", right_on="nombre", threshold=0.7)
        assert len(result) > 0

    def test_left_join(self) -> None:
        left = pd.DataFrame({"name": ["Jon", "ZZZZZ"]})
        right = pd.DataFrame({"name": ["John"]})
        result = fuzzy_join(left, right, on="name", threshold=0.7, how="left")
        assert len(result) == 2  # Both left rows kept

    def test_no_matches(self) -> None:
        left = pd.DataFrame({"name": ["abc"]})
        right = pd.DataFrame({"name": ["xyz"]})
        result = fuzzy_join(left, right, on="name", threshold=0.99)
        assert len(result) == 0

    def test_limit(self) -> None:
        left = pd.DataFrame({"name": ["Jon"]})
        right = pd.DataFrame({"name": ["John", "Jonathan", "Jonas"]})
        result = fuzzy_join(left, right, on="name", threshold=0.5, limit=3)
        assert len(result) >= 1

    def test_invalid_how(self) -> None:
        left = pd.DataFrame({"name": ["Jon"]})
        right = pd.DataFrame({"name": ["John"]})
        with pytest.raises(ValueError, match="how must be"):
            fuzzy_join(left, right, on="name", how="outer")

    def test_column_spec_errors(self) -> None:
        left = pd.DataFrame({"name": ["Jon"]})
        right = pd.DataFrame({"name": ["John"]})
        with pytest.raises(ValueError, match="Cannot specify both"):
            fuzzy_join(left, right, on="name", left_on="name")
        with pytest.raises(ValueError, match="Must specify"):
            fuzzy_join(left, right, left_on="name")

    def test_different_scorers(self) -> None:
        left = pd.DataFrame({"name": ["Jon"]})
        right = pd.DataFrame({"name": ["John"]})
        for scorer in ["jaro_winkler", "levenshtein", "cosine"]:
            result = fuzzy_join(left, right, on="name", scorer=scorer, threshold=0.3)
            assert "_score" in result.columns
