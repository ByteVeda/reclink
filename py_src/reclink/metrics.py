"""String similarity and distance metrics.

All metrics are implemented in Rust for maximum performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reclink._core import (
    cdist as _cdist,
)
from reclink._core import (
    cosine as _cosine,
)
from reclink._core import (
    damerau_levenshtein as _damerau_levenshtein,
)
from reclink._core import (
    damerau_levenshtein_similarity as _damerau_levenshtein_similarity,
)
from reclink._core import (
    hamming as _hamming,
)
from reclink._core import (
    hamming_similarity as _hamming_similarity,
)
from reclink._core import (
    jaccard as _jaccard,
)
from reclink._core import (
    jaro as _jaro,
)
from reclink._core import (
    jaro_winkler as _jaro_winkler,
)
from reclink._core import (
    levenshtein as _levenshtein,
)
from reclink._core import (
    levenshtein_similarity as _levenshtein_similarity,
)
from reclink._core import (
    sorensen_dice as _sorensen_dice,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


def levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    int
        Minimum number of single-character edits.

    Examples
    --------
    >>> levenshtein("kitten", "sitting")
    3
    """
    return _levenshtein(a, b)


def levenshtein_similarity(a: str, b: str) -> float:
    """Compute normalized Levenshtein similarity in [0, 1].

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score where 1.0 means identical.
    """
    return _levenshtein_similarity(a, b)


def damerau_levenshtein(a: str, b: str) -> int:
    """Compute Damerau-Levenshtein distance (includes transpositions).

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    int
        Edit distance including transpositions.
    """
    return _damerau_levenshtein(a, b)


def damerau_levenshtein_similarity(a: str, b: str) -> float:
    """Compute normalized Damerau-Levenshtein similarity in [0, 1].

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score where 1.0 means identical.
    """
    return _damerau_levenshtein_similarity(a, b)


def hamming(a: str, b: str) -> int:
    """Compute the Hamming distance between two equal-length strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string (must be same length as `a`).

    Returns
    -------
    int
        Number of positions where characters differ.

    Raises
    ------
    ValueError
        If strings have different lengths.
    """
    return _hamming(a, b)


def hamming_similarity(a: str, b: str) -> float:
    """Compute normalized Hamming similarity in [0, 1].

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string (must be same length as `a`).

    Returns
    -------
    float
        Similarity score where 1.0 means identical.

    Raises
    ------
    ValueError
        If strings have different lengths.
    """
    return _hamming_similarity(a, b)


def jaro(a: str, b: str) -> float:
    """Compute the Jaro similarity between two strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score in [0, 1].

    Examples
    --------
    >>> round(jaro("martha", "marhta"), 4)
    0.9444
    """
    return _jaro(a, b)


def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1) -> float:
    """Compute the Jaro-Winkler similarity between two strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    prefix_weight : float, optional
        Scaling factor for common prefix bonus (default 0.1, max 0.25).

    Returns
    -------
    float
        Similarity score in [0, 1].

    Examples
    --------
    >>> round(jaro_winkler("Jon Smith", "John Smyth"), 3)
    0.832
    """
    return _jaro_winkler(a, b, prefix_weight)


def cosine(a: str, b: str, n: int = 2) -> float:
    """Compute cosine similarity between character n-gram vectors.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    n : int, optional
        N-gram size (default 2, bigrams).

    Returns
    -------
    float
        Cosine similarity in [0, 1].
    """
    return _cosine(a, b, n)


def jaccard(a: str, b: str) -> float:
    """Compute Jaccard similarity between whitespace-tokenized sets.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Jaccard index in [0, 1].
    """
    return _jaccard(a, b)


def sorensen_dice(a: str, b: str) -> float:
    """Compute the Sorensen-Dice coefficient between character bigrams.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Dice coefficient in [0, 1].
    """
    return _sorensen_dice(a, b)


def cdist(
    a: Sequence[str],
    b: Sequence[str],
    scorer: str = "jaro_winkler",
    workers: int | None = None,
) -> NDArray[np.float64]:
    """Compute pairwise similarity matrix between two lists of strings.

    Parameters
    ----------
    a : sequence of str
        First list of strings.
    b : sequence of str
        Second list of strings.
    scorer : str, optional
        Metric name (default "jaro_winkler"). One of: levenshtein,
        damerau_levenshtein, hamming, jaro, jaro_winkler, cosine,
        jaccard, sorensen_dice.
    workers : int or None, optional
        Number of parallel threads (default: all cores).

    Returns
    -------
    numpy.ndarray
        2D array of shape (len(a), len(b)) with similarity scores.

    Examples
    --------
    >>> matrix = cdist(["Jon", "Jane"], ["John", "Janet"], scorer="jaro_winkler")
    >>> matrix.shape
    (2, 2)
    """
    return _cdist(list(a), list(b), scorer, workers)
