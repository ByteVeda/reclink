"""Utility helpers for reclink.

Provides input-validation and diagnostic helpers that document
library conventions for edge cases.
"""

from __future__ import annotations


def validate_strings(a: str, b: str) -> tuple[str, str, str]:
    """Check two input strings and return a status describing edge cases.

    Parameters
    ----------
    a : str
        Left input string.
    b : str
        Right input string.

    Returns
    -------
    tuple of (str, str, str)
        ``(a, b, status)`` where *status* is one of:

        - ``"ok"`` -- both strings are non-empty.
        - ``"both_empty"`` -- both strings are empty.
        - ``"left_empty"`` -- only *a* is empty.
        - ``"right_empty"`` -- only *b* is empty.

    Notes
    -----
    Library conventions for empty strings:

    * **Distance metrics**: empty vs non-empty returns the length of
      the non-empty string; both empty returns 0.
    * **Similarity metrics**: both empty returns 1.0;
      one empty returns 0.0.
    * **Phonetic**: soundex returns ``"0000"``; metaphone returns ``""``.
    """
    if not a and not b:
        return a, b, "both_empty"
    if not a:
        return a, b, "left_empty"
    if not b:
        return a, b, "right_empty"
    return a, b, "ok"
