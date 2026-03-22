"""Phonetic encoding algorithms.

All algorithms are implemented in Rust for maximum performance.
"""

from __future__ import annotations

from reclink._core import (
    caverphone as _caverphone,
)
from reclink._core import (
    cologne_phonetic as _cologne_phonetic,
)
from reclink._core import (
    daitch_mokotoff as _daitch_mokotoff,
)
from reclink._core import (
    double_metaphone as _double_metaphone,
)
from reclink._core import (
    metaphone as _metaphone,
)
from reclink._core import (
    mra as _mra,
)
from reclink._core import (
    mra_compare as _mra_compare,
)
from reclink._core import (
    nysiis as _nysiis,
)
from reclink._core import (
    phonex as _phonex,
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


def caverphone(s: str) -> str:
    """Compute the Caverphone 2 code for a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        10-character Caverphone 2 code.
    """
    return _caverphone(s)


def cologne_phonetic(s: str) -> str:
    """Compute the Cologne Phonetic (Kolner Phonetik) code for a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Variable-length digit string.
    """
    return _cologne_phonetic(s)


def phonex(s: str) -> str:
    """Compute the Phonex (improved Soundex) code for a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        4-character Phonex code.

    Examples
    --------
    >>> phonex("Smith")
    'S530'
    """
    return _phonex(s)


def mra(s: str) -> str:
    """Compute the Match Rating Approach code for a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        MRA phonetic code (up to 6 characters).

    Examples
    --------
    >>> mra("Smith")
    'SMTH'
    """
    return _mra(s)


def mra_compare(a: str, b: str) -> bool:
    """Compare two strings using the Match Rating Approach.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    bool
        True if the strings are considered a phonetic match.

    Examples
    --------
    >>> mra_compare("Smith", "Smyth")
    True
    """
    return _mra_compare(a, b)


def daitch_mokotoff(s: str) -> str:
    """Compute the Daitch-Mokotoff Soundex code for a string.

    Produces 6-digit numeric codes for Slavic, Germanic, and Hebrew
    names. Returns comma-separated codes when multiple alternatives exist.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Comma-separated 6-digit codes.

    Examples
    --------
    >>> daitch_mokotoff("Schwartz")  # doctest: +SKIP
    '479400'
    """
    return _daitch_mokotoff(s)
