"""Tests for reclink.utils."""

from __future__ import annotations

from reclink.utils import validate_strings


class TestValidateStrings:
    def test_both_non_empty(self) -> None:
        a, b, status = validate_strings("hello", "world")
        assert a == "hello"
        assert b == "world"
        assert status == "ok"

    def test_both_empty(self) -> None:
        _, _, status = validate_strings("", "")
        assert status == "both_empty"

    def test_left_empty(self) -> None:
        _, _, status = validate_strings("", "world")
        assert status == "left_empty"

    def test_right_empty(self) -> None:
        _, _, status = validate_strings("hello", "")
        assert status == "right_empty"
