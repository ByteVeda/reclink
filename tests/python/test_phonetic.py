"""Tests for phonetic encoding algorithms."""

import reclink


class TestSoundex:
    def test_known_values(self) -> None:
        assert reclink.soundex("Robert") == "R163"
        assert reclink.soundex("Rupert") == "R163"

    def test_smith_variants(self) -> None:
        assert reclink.soundex("Smith") == "S530"
        assert reclink.soundex("Smyth") == "S530"

    def test_empty(self) -> None:
        assert reclink.soundex("") == "0000"


class TestMetaphone:
    def test_basic(self) -> None:
        code = reclink.metaphone("Smith")
        assert len(code) > 0

    def test_empty(self) -> None:
        assert reclink.metaphone("") == ""


class TestDoubleMetaphone:
    def test_returns_tuple(self) -> None:
        primary, alternate = reclink.double_metaphone("Smith")
        assert isinstance(primary, str)
        assert isinstance(alternate, str)


class TestNysiis:
    def test_known_values(self) -> None:
        assert reclink.nysiis("Johnson") == "JANSAN"

    def test_empty(self) -> None:
        assert reclink.nysiis("") == ""
