"""Optional pandas/polars compatibility utilities."""

from __future__ import annotations

from typing import Any


def results_to_dataframe(results: list[dict[str, Any]]) -> Any:
    """Convert match results to a pandas DataFrame if available.

    Parameters
    ----------
    results : list of dict
        Match results from pipeline.dedup() or pipeline.link().

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: left_id, right_id, score, scores.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    import pandas as pd

    return pd.DataFrame(results)


def results_to_polars(results: list[dict[str, Any]]) -> Any:
    """Convert match results to a polars DataFrame if available.

    Parameters
    ----------
    results : list of dict
        Match results from pipeline.dedup() or pipeline.link().

    Returns
    -------
    polars.DataFrame
        DataFrame with columns: left_id, right_id, score, scores.

    Raises
    ------
    ImportError
        If polars is not installed.
    """
    import polars as pl

    return pl.DataFrame(results)
