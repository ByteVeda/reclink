"""Scoring presets for common record linkage use cases.

Each factory returns a pre-configured ``CompositeScorer`` ready for use.
"""

from reclink._core import PyCompositeScorer as CompositeScorer


def name_matching() -> CompositeScorer:
    """Scorer tuned for person-name matching.

    Weights
    -------
    jaro_winkler : 0.5
    token_sort : 0.3
    phonetic_hybrid : 0.2
    """
    return CompositeScorer.preset("name_matching")


def address_matching() -> CompositeScorer:
    """Scorer tuned for address matching.

    Weights
    -------
    token_set : 0.5
    jaccard : 0.3
    levenshtein : 0.2
    """
    return CompositeScorer.preset("address_matching")


def general_purpose() -> CompositeScorer:
    """General-purpose composite scorer.

    Weights
    -------
    jaro_winkler : 0.4
    cosine : 0.4
    token_sort : 0.2
    """
    return CompositeScorer.preset("general_purpose")
