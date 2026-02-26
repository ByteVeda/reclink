"""Tests for string similarity metrics."""

import numpy as np
import pytest

import reclink


class TestLevenshtein:
    def test_identical(self) -> None:
        assert reclink.levenshtein("hello", "hello") == 0

    def test_known_values(self) -> None:
        assert reclink.levenshtein("kitten", "sitting") == 3
        assert reclink.levenshtein("saturday", "sunday") == 3

    def test_empty(self) -> None:
        assert reclink.levenshtein("", "") == 0
        assert reclink.levenshtein("abc", "") == 3

    def test_similarity(self) -> None:
        sim = reclink.levenshtein_similarity("kitten", "sitting")
        assert 0.5 < sim < 0.7


class TestDamerauLevenshtein:
    def test_transposition(self) -> None:
        assert reclink.damerau_levenshtein("ab", "ba") == 1

    def test_known_values(self) -> None:
        assert reclink.damerau_levenshtein("ca", "abc") == 3


class TestHamming:
    def test_known_values(self) -> None:
        assert reclink.hamming("karolin", "kathrin") == 3

    def test_unequal_length(self) -> None:
        with pytest.raises(ValueError):
            reclink.hamming("abc", "ab")


class TestJaro:
    def test_known_values(self) -> None:
        assert abs(reclink.jaro("martha", "marhta") - 0.9444) < 0.001

    def test_identical(self) -> None:
        assert reclink.jaro("hello", "hello") == 1.0


class TestJaroWinkler:
    def test_known_values(self) -> None:
        assert abs(reclink.jaro_winkler("martha", "marhta") - 0.9611) < 0.001

    def test_prefix_boost(self) -> None:
        jw = reclink.jaro_winkler("abc", "abx")
        j = reclink.jaro("abc", "abx")
        assert jw >= j


class TestCosine:
    def test_identical(self) -> None:
        assert abs(reclink.cosine("hello", "hello") - 1.0) < 0.001

    def test_different(self) -> None:
        assert abs(reclink.cosine("ab", "cd")) < 0.001


class TestJaccard:
    def test_identical(self) -> None:
        assert abs(reclink.jaccard("hello world", "hello world") - 1.0) < 0.001

    def test_partial(self) -> None:
        sim = reclink.jaccard("cat dog", "cat bird")
        assert abs(sim - 1 / 3) < 0.001


class TestSorensenDice:
    def test_known_values(self) -> None:
        assert abs(reclink.sorensen_dice("night", "nacht") - 0.25) < 0.001


class TestCdist:
    def test_basic(self) -> None:
        a = ["Jon", "Jane"]
        b = ["John", "Janet"]
        matrix = reclink.cdist(a, b, scorer="jaro_winkler")
        assert matrix.shape == (2, 2)
        assert matrix.dtype == np.float64
        assert 0 <= matrix[0, 0] <= 1
        assert 0 <= matrix[1, 1] <= 1

    def test_different_scorers(self) -> None:
        a = ["hello"]
        b = ["hello", "world"]
        for scorer in ["jaro", "jaro_winkler", "cosine", "jaccard", "sorensen_dice"]:
            matrix = reclink.cdist(a, b, scorer=scorer)
            assert matrix.shape == (1, 2)
