"""Phonetic encoding algorithms.

All algorithms are implemented in Rust for maximum performance.
"""

from __future__ import annotations

from reclink._core import (
    double_metaphone as _double_metaphone,
)
from reclink._core import (
    metaphone as _metaphone,
)
from reclink._core import (
    nysiis as _nysiis,
)
from reclink._core import (
    soundex as _soundex,
)


def soundex(s: str) -> str:
    """Compute the Soundex code for a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        4-character Soundex code (e.g., "S530" for "Smith").

    Examples
    --------
    >>> soundex("Smith")
    'S530'
    >>> soundex("Smyth")
    'S530'
    """
    return _soundex(s)


def metaphone(s: str) -> str:
    """Compute the Metaphone code for a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Variable-length Metaphone code.
    """
    return _metaphone(s)


def double_metaphone(s: str) -> tuple[str, str]:
    """Compute Double Metaphone codes for a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    tuple[str, str]
        Primary and alternate phonetic codes.
    """
    return _double_metaphone(s)


def nysiis(s: str) -> str:
    """Compute the NYSIIS code for a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        NYSIIS phonetic code (up to 6 characters).
    """
    return _nysiis(s)
