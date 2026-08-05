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
    damerau_levenshtein_threshold as _damerau_levenshtein_threshold,
)
from reclink._core import (
    gotoh as _gotoh,
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
    lcs_length as _lcs_length,
)
from reclink._core import (
    lcs_similarity as _lcs_similarity,
)
from reclink._core import (
    levenshtein as _levenshtein,
)
from reclink._core import (
    levenshtein_similarity as _levenshtein_similarity,
)
from reclink._core import (
    levenshtein_threshold as _levenshtein_threshold,
)
from reclink._core import (
    longest_common_substring_length as _longest_common_substring_length,
)
from reclink._core import (
    longest_common_substring_similarity as _longest_common_substring_similarity,
)
from reclink._core import (
    match_batch as _match_batch,
)
from reclink._core import (
    match_best as _match_best,
)
from reclink._core import (
    monge_elkan as _monge_elkan,
)
from reclink._core import (
    needleman_wunsch as _needleman_wunsch,
)
from reclink._core import (
    ngram_similarity as _ngram_similarity,
)
from reclink._core import (
    partial_ratio as _partial_ratio,
)
from reclink._core import (
    phonetic_hybrid as _phonetic_hybrid,
)
from reclink._core import (
    ratcliff_obershelp as _ratcliff_obershelp,
)
from reclink._core import (
    smith_waterman as _smith_waterman,
)
from reclink._core import (
    smith_waterman_similarity as _smith_waterman_similarity,
)
from reclink._core import (
    sorensen_dice as _sorensen_dice,
)
from reclink._core import (
    token_set_ratio as _token_set_ratio,
)
from reclink._core import (
    token_sort_ratio as _token_sort_ratio,
)
from reclink._core import (
    weighted_levenshtein as _weighted_levenshtein,
)
from reclink._core import (
    weighted_levenshtein_similarity as _weighted_levenshtein_similarity,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from reclink._core import PhoneticAlgorithm, Scorer


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


def levenshtein_threshold(a: str, b: str, max_distance: int) -> int | None:
    """Compute Levenshtein distance with early termination.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    max_distance : int
        Maximum distance threshold. If the actual distance exceeds this,
        returns None instead of computing the full distance.

    Returns
    -------
    int or None
        The edit distance if <= max_distance, otherwise None.

    Examples
    --------
    >>> levenshtein_threshold("kitten", "sitting", 3)
    3
    >>> levenshtein_threshold("kitten", "sitting", 2) is None
    True
    """
    return _levenshtein_threshold(a, b, max_distance)


def damerau_levenshtein_threshold(a: str, b: str, max_distance: int) -> int | None:
    """Compute Damerau-Levenshtein distance with early termination.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    max_distance : int
        Maximum distance threshold. If the actual distance exceeds this,
        returns None instead of computing the full distance.

    Returns
    -------
    int or None
        The edit distance if <= max_distance, otherwise None.

    Examples
    --------
    >>> damerau_levenshtein_threshold("ab", "ba", 1)
    1
    >>> damerau_levenshtein_threshold("abc", "xyz", 1) is None
    True
    """
    return _damerau_levenshtein_threshold(a, b, max_distance)


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


def weighted_levenshtein(
    a: str,
    b: str,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
    substitute_cost: float = 1.0,
    transpose_cost: float = 1.0,
) -> float:
    """Compute weighted edit distance with configurable operation costs.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    insert_cost : float, optional
        Cost of character insertion (default 1.0).
    delete_cost : float, optional
        Cost of character deletion (default 1.0).
    substitute_cost : float, optional
        Cost of character substitution (default 1.0).
    transpose_cost : float, optional
        Cost of adjacent character transposition (default 1.0).

    Returns
    -------
    float
        Weighted edit distance.
    """
    return _weighted_levenshtein(a, b, insert_cost, delete_cost, substitute_cost, transpose_cost)


def weighted_levenshtein_similarity(
    a: str,
    b: str,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
    substitute_cost: float = 1.0,
    transpose_cost: float = 1.0,
) -> float:
    """Compute normalized weighted Levenshtein similarity in [0, 1].

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    insert_cost : float, optional
        Cost of character insertion (default 1.0).
    delete_cost : float, optional
        Cost of character deletion (default 1.0).
    substitute_cost : float, optional
        Cost of character substitution (default 1.0).
    transpose_cost : float, optional
        Cost of adjacent character transposition (default 1.0).

    Returns
    -------
    float
        Similarity score where 1.0 means identical.
    """
    return _weighted_levenshtein_similarity(
        a, b, insert_cost, delete_cost, substitute_cost, transpose_cost
    )


def token_sort_ratio(a: str, b: str) -> float:
    """Compute token sort ratio between two strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score in [0, 1] after sorting tokens alphabetically.

    Examples
    --------
    >>> round(token_sort_ratio("John Smith", "Smith John"), 1)
    1.0
    """
    return _token_sort_ratio(a, b)


def token_set_ratio(a: str, b: str) -> float:
    """Compute token set ratio between two strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score in [0, 1] using set intersection/remainder logic.
    """
    return _token_set_ratio(a, b)


def partial_ratio(a: str, b: str) -> float:
    """Compute partial ratio (best substring match) between two strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Best substring match similarity in [0, 1].

    Examples
    --------
    >>> round(partial_ratio("test", "this is a test"), 1)
    1.0
    """
    return _partial_ratio(a, b)


def lcs_length(a: str, b: str) -> int:
    """Compute the length of the longest common subsequence.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    int
        Length of the longest common subsequence.
    """
    return _lcs_length(a, b)


def lcs_similarity(a: str, b: str) -> float:
    """Compute normalized LCS similarity in [0, 1].

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score: 2 * lcs_len / (len_a + len_b).
    """
    return _lcs_similarity(a, b)


def longest_common_substring_length(a: str, b: str) -> int:
    """Compute the length of the longest common substring.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    int
        Length of the longest contiguous common substring.
    """
    return _longest_common_substring_length(a, b)


def longest_common_substring_similarity(a: str, b: str) -> float:
    """Compute normalized longest common substring similarity in [0, 1].

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score: 2 * substr_len / (len_a + len_b).
    """
    return _longest_common_substring_similarity(a, b)


def ngram_similarity(a: str, b: str, n: int = 2) -> float:
    """Compute n-gram Jaccard similarity between two strings.

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
        Jaccard coefficient over character n-gram sets in [0, 1].
    """
    return _ngram_similarity(a, b, n)


def smith_waterman(
    a: str,
    b: str,
    match_score: float = 2.0,
    mismatch_penalty: float = -1.0,
    gap_penalty: float = -1.0,
) -> float:
    """Compute raw Smith-Waterman local alignment score.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    match_score : float, optional
        Score for a character match (default 2.0).
    mismatch_penalty : float, optional
        Penalty for a mismatch (default -1.0).
    gap_penalty : float, optional
        Penalty for a gap/indel (default -1.0).

    Returns
    -------
    float
        Raw alignment score (non-negative).
    """
    return _smith_waterman(a, b, match_score, mismatch_penalty, gap_penalty)


def smith_waterman_similarity(a: str, b: str) -> float:
    """Compute normalized Smith-Waterman similarity in [0, 1].

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Normalized local alignment similarity.
    """
    return _smith_waterman_similarity(a, b)


def phonetic_hybrid(
    a: str,
    b: str,
    phonetic: PhoneticAlgorithm = "soundex",
    metric: Scorer = "jaro_winkler",
    phonetic_weight: float = 0.3,
) -> float:
    """Compute phonetic + edit distance hybrid similarity.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    phonetic : str, optional
        Phonetic algorithm name (default "soundex").
    metric : str, optional
        Edit distance metric name (default "jaro_winkler").
    phonetic_weight : float, optional
        Weight for phonetic component (default 0.3).

    Returns
    -------
    float
        Weighted hybrid similarity in [0, 1].
    """
    return _phonetic_hybrid(a, b, phonetic, metric, phonetic_weight)


def match_best(
    query: str,
    candidates: list[str],
    scorer: Scorer = "jaro_winkler",
    threshold: float | None = None,
    workers: int | None = None,
) -> tuple[str, float, int] | None:
    """Find the best match for a query among candidates.

    Parameters
    ----------
    query : str
        The string to match.
    candidates : list of str
        Candidate strings to compare against.
    scorer : str, optional
        Metric name (default "jaro_winkler").
    threshold : float or None, optional
        Minimum similarity score to return a result (default None).
    workers : int or None, optional
        Number of parallel threads (default: all cores).

    Returns
    -------
    tuple of (str, float, int) or None
        A tuple of (matched_string, score, index) or None if no match.

    Examples
    --------
    >>> match_best("hello", ["hallo", "world", "help"])
    ('hallo', ..., 0)
    """
    return _match_best(query, candidates, scorer, threshold, workers)


def match_batch(
    query: str,
    candidates: list[str],
    scorer: Scorer = "jaro_winkler",
    threshold: float | None = None,
    limit: int | None = None,
    workers: int | None = None,
) -> list[tuple[str, float, int]]:
    """Find all matches for a query among candidates, sorted by descending score.

    Parameters
    ----------
    query : str
        The string to match.
    candidates : list of str
        Candidate strings to compare against.
    scorer : str, optional
        Metric name (default "jaro_winkler").
    threshold : float or None, optional
        Minimum similarity score to include (default None).
    limit : int or None, optional
        Maximum number of results to return (default None).
    workers : int or None, optional
        Number of parallel threads (default: all cores).

    Returns
    -------
    list of tuple of (str, float, int)
        List of (matched_string, score, index) tuples sorted by descending score.

    Examples
    --------
    >>> match_batch("hello", ["hallo", "world", "help"], limit=2)
    [('hallo', ..., 0), ('help', ..., 2)]
    """
    return _match_batch(query, candidates, scorer, threshold, limit, workers)


def cdist(
    a: Sequence[str],
    b: Sequence[str],
    scorer: Scorer = "jaro_winkler",
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
        Metric name (default "jaro_winkler"). Supports all metric names
        including token_sort, token_set, partial_ratio, lcs, etc.
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


def ratcliff_obershelp(a: str, b: str) -> float:
    """Compute Ratcliff-Obershelp (Gestalt Pattern Matching) similarity.

    Recursively finds the longest common substring, then recursively
    matches the remaining left and right portions.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score in [0, 1], where 1 means identical.

    Examples
    --------
    >>> ratcliff_obershelp("abcde", "abdce")
    0.8
    """
    return _ratcliff_obershelp(a, b)


def needleman_wunsch(a: str, b: str) -> float:
    """Compute Needleman-Wunsch global alignment similarity.

    Uses dynamic programming for global sequence alignment with
    default parameters (match=2, mismatch=-1, gap=-1).

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score in [0, 1], where 1 means identical.

    Examples
    --------
    >>> needleman_wunsch("kitten", "sitting")
    0.5
    """
    return _needleman_wunsch(a, b)


def gotoh(a: str, b: str) -> float:
    """Compute Gotoh (affine gap penalty) global alignment similarity.

    Extends Needleman-Wunsch with separate gap-open and gap-extend
    costs, better modeling biological insertions/deletions.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    float
        Similarity score in [0, 1], where 1 means identical.

    Examples
    --------
    >>> gotoh("kitten", "sitting")
    0.4
    """
    return _gotoh(a, b)


def monge_elkan(
    a: str,
    b: str,
    inner_metric: Scorer | None = None,
) -> float:
    """Compute Monge-Elkan token-based similarity.

    For each token in ``a``, finds the best-matching token in ``b``
    using the inner metric. Returns the average of best matches.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.
    inner_metric : str or None, optional
        Name of the inner metric for token comparison.
        Default is ``"jaro_winkler"``.

    Returns
    -------
    float
        Similarity score in [0, 1], where 1 means identical.

    Examples
    --------
    >>> monge_elkan("john smith", "smith john")
    1.0
    """
    return _monge_elkan(a, b, inner_metric)
