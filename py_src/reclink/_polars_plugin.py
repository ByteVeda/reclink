"""Native Polars expression plugin for reclink.

Provides zero-GIL-overhead string similarity, phonetic encoding, and
match_best expressions that operate directly on Arrow arrays.

Requires building with ``--features polars-plugin``.

Usage
-----
>>> import polars as pl
>>> from reclink._polars_plugin import similarity, phonetic, match_best
>>>
>>> df = pl.DataFrame({"a": ["John", "Jane"], "b": ["Jon", "Janet"]})
>>> df.with_columns(similarity(pl.col("a"), pl.col("b"), scorer="jaro_winkler"))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

_PLUGIN_PATH = Path(__file__).parent


def similarity(
    a: pl.Expr,
    b: pl.Expr,
    *,
    scorer: str = "jaro_winkler",
) -> pl.Expr:
    """Compute string similarity between two columns.

    Parameters
    ----------
    a : pl.Expr
        Left string column.
    b : pl.Expr
        Right string column.
    scorer : str
        Similarity metric name (e.g., ``"jaro_winkler"``, ``"levenshtein"``).

    Returns
    -------
    pl.Expr
        Float64 column of similarity scores.
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="reclink_similarity",
        args=[a, b],
        kwargs={"scorer": scorer},
        is_elementwise=True,
    )


def phonetic(
    expr: pl.Expr,
    *,
    algorithm: str = "soundex",
) -> pl.Expr:
    """Phonetic encoding of a string column.

    Parameters
    ----------
    expr : pl.Expr
        String column to encode.
    algorithm : str
        Phonetic algorithm (e.g., ``"soundex"``, ``"metaphone"``,
        ``"beider_morse"``).

    Returns
    -------
    pl.Expr
        String column of phonetic codes.
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="reclink_phonetic",
        args=[expr],
        kwargs={"algorithm": algorithm},
        is_elementwise=True,
    )


def match_best(
    expr: pl.Expr,
    candidates: list[str],
    *,
    scorer: str = "jaro_winkler",
    threshold: float = 0.0,
) -> pl.Expr:
    """Find the best matching candidate for each value.

    Parameters
    ----------
    expr : pl.Expr
        String column to match.
    candidates : list of str
        Candidate strings to match against.
    scorer : str
        Similarity metric name.
    threshold : float
        Minimum score threshold. Values below this return null.

    Returns
    -------
    pl.Expr
        String column of best matches (null if below threshold).
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="reclink_match_best",
        args=[expr],
        kwargs={
            "candidates": candidates,
            "scorer": scorer,
            "threshold": threshold,
        },
        is_elementwise=True,
    )
