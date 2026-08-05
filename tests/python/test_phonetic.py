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


class TestPhonex:
    def test_smith(self) -> None:
        assert reclink.phonex("Smith") == "S530"
        assert reclink.phonex("Smyth") == "S530"

    def test_empty(self) -> None:
        assert reclink.phonex("") == "0000"

    def test_prefix_kn(self) -> None:
        code = reclink.phonex("Knight")
        assert code.startswith("N")


class TestMRA:
    def test_encode(self) -> None:
        assert reclink.mra("Smith") == "SMTH"

    def test_compare_similar(self) -> None:
        assert reclink.mra_compare("Smith", "Smyth")

    def test_compare_different(self) -> None:
        assert not reclink.mra_compare("Smith", "Jones")

    def test_empty(self) -> None:
        assert reclink.mra("") == ""


class TestDaitchMokotoff:
    def test_basic(self) -> None:
        code = reclink.daitch_mokotoff("Schwartz")
        # Should be 6-digit codes
        for c in code.split(","):
            assert len(c) == 6
            assert c.isdigit()

    def test_empty(self) -> None:
        assert reclink.daitch_mokotoff("") == "000000"

    def test_branching(self) -> None:
        # CH rules produce multiple codes
        code = reclink.daitch_mokotoff("Chaim")
        assert "," in code  # Multiple codes from branching
