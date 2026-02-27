"""Tests for Pandas accessor integration."""

import pytest

pd = pytest.importorskip("pandas")

import reclink  # noqa: E402


class TestSeriesAccessor:
    def test_match_best(self) -> None:
        s = pd.Series(["John", "Jane", "Bob"])
        candidates = ["Jon", "Janet", "Robert"]
        result = s.reclink.match_best(candidates)
        assert len(result) == 3
        assert result.iloc[0] is not None
        assert result.iloc[0][0] == "Jon"

    def test_phonetic(self) -> None:
        s = pd.Series(["Smith", "Johnson"])
        result = s.reclink.phonetic("soundex")
        assert len(result) == 2
        assert result.iloc[0] == reclink.soundex("Smith")

    def test_deduplicate(self) -> None:
        s = pd.Series(["John Smith", "Jon Smith", "Jane Doe", "Janet Doe"])
        groups = s.reclink.deduplicate(threshold=0.8)
        assert len(groups) >= 1
        # John Smith and Jon Smith should be grouped
        found = any(0 in g and 1 in g for g in groups)
        assert found


class TestDataFrameAccessor:
    def test_fuzzy_merge(self) -> None:
        left = pd.DataFrame({"name": ["John", "Jane"], "age": [30, 25]})
        right = pd.DataFrame({"name": ["Jon", "Janet"], "city": ["NYC", "LA"]})
        result = left.reclink.fuzzy_merge(right, left_on="name", right_on="name", threshold=0.7)
        assert len(result) >= 1
        assert "_score" in result.columns

    def test_fuzzy_merge_no_matches(self) -> None:
        left = pd.DataFrame({"name": ["abc"]})
        right = pd.DataFrame({"name": ["xyz"]})
        result = left.reclink.fuzzy_merge(right, left_on="name", right_on="name", threshold=0.99)
        assert len(result) == 0
