"""Fuzzy join API for DataFrame-level fuzzy matching.

Provides a standalone ``fuzzy_join()`` function that works with both
pandas and polars DataFrames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import reclink

if TYPE_CHECKING:
    from reclink._core import Scorer


def fuzzy_join(
    left: Any,
    right: Any,
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    scorer: Scorer = "jaro_winkler",
    threshold: float = 0.85,
    how: str = "inner",
    limit: int = 1,
) -> Any:
    """Fuzzy join two DataFrames on string columns.

    Parameters
    ----------
    left : DataFrame
        Left DataFrame (pandas or polars).
    right : DataFrame
        Right DataFrame (pandas or polars).
    on : str or None, optional
        Column name to join on (same in both DataFrames).
        Mutually exclusive with ``left_on``/``right_on``.
    left_on : str or None, optional
        Column name in the left DataFrame.
    right_on : str or None, optional
        Column name in the right DataFrame.
    scorer : str, optional
        Metric name (default ``"jaro_winkler"``).
    threshold : float, optional
        Minimum similarity to include a match (default 0.85).
    how : str, optional
        Join type: ``"inner"`` (only matches) or ``"left"``
        (keep all left rows, nulls for non-matches). Default ``"inner"``.
    limit : int, optional
        Maximum number of matches per left row (default 1).

    Returns
    -------
    DataFrame
        Merged DataFrame with a ``_score`` column. Right columns are
        suffixed with ``_right``.

    Raises
    ------
    ValueError
        If column specification is invalid or ``how`` is unknown.

    Examples
    --------
    >>> import pandas as pd
    >>> left = pd.DataFrame({"name": ["Jon", "Jane"]})
    >>> right = pd.DataFrame({"name": ["John", "Janet"]})
    >>> result = fuzzy_join(left, right, on="name", threshold=0.8)
    """
    left_col, right_col = _resolve_columns(on, left_on, right_on)

    if how not in ("inner", "left"):
        msg = f"how must be 'inner' or 'left', got '{how}'"
        raise ValueError(msg)

    if _is_pandas(left):
        return _fuzzy_join_pandas(left, right, left_col, right_col, scorer, threshold, how, limit)
    if _is_polars(left):
        return _fuzzy_join_polars(left, right, left_col, right_col, scorer, threshold, how, limit)

    msg = f"Unsupported DataFrame type: {type(left).__name__}"
    raise TypeError(msg)


def _resolve_columns(
    on: str | None,
    left_on: str | None,
    right_on: str | None,
) -> tuple[str, str]:
    if on is not None:
        if left_on is not None or right_on is not None:
            msg = "Cannot specify both 'on' and 'left_on'/'right_on'"
            raise ValueError(msg)
        return on, on
    if left_on is None or right_on is None:
        msg = "Must specify either 'on' or both 'left_on' and 'right_on'"
        raise ValueError(msg)
    return left_on, right_on


def _is_pandas(obj: Any) -> bool:
    try:
        import pandas as pd

        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False


def _is_polars(obj: Any) -> bool:
    try:
        import polars as pl

        return isinstance(obj, (pl.DataFrame, pl.LazyFrame))
    except ImportError:
        return False


def _fuzzy_join_pandas(
    left: Any,
    right: Any,
    left_col: str,
    right_col: str,
    scorer: Scorer,
    threshold: float,
    how: str,
    limit: int,
) -> Any:
    import pandas as pd

    right_values = right[right_col].astype(str).tolist()
    matches: list[dict[str, Any]] = []

    for left_idx, left_val in zip(left.index, left[left_col].astype(str), strict=True):
        if limit == 1:
            result = reclink.match_best(left_val, right_values, scorer, threshold)
            if result is not None:
                _, score, right_idx = result
                matches.append(
                    {
                        "_left_idx": left_idx,
                        "_right_idx": right.index[right_idx],
                        "_score": score,
                    }
                )
            elif how == "left":
                matches.append({"_left_idx": left_idx, "_right_idx": None, "_score": None})
        else:
            results = reclink.match_batch(left_val, right_values, scorer, threshold, limit)
            if results:
                for _matched_str, score, right_idx in results:
                    matches.append(
                        {
                            "_left_idx": left_idx,
                            "_right_idx": right.index[right_idx],
                            "_score": score,
                        }
                    )
            elif how == "left":
                matches.append({"_left_idx": left_idx, "_right_idx": None, "_score": None})

    if not matches:
        return pd.DataFrame()

    match_df = pd.DataFrame(matches)

    # Build result
    left_matched = left.loc[match_df["_left_idx"].values].reset_index(drop=True)

    # Handle right side (may have None indices for left joins)
    valid_right = match_df["_right_idx"].notna()
    right_rows = []
    for is_valid, ridx in zip(valid_right, match_df["_right_idx"], strict=True):
        if is_valid:
            right_rows.append(right.loc[ridx])
        else:
            right_rows.append(pd.Series({c: None for c in right.columns}, dtype=object))

    right_matched = pd.DataFrame(right_rows).reset_index(drop=True)
    right_matched.columns = [f"{c}_right" for c in right_matched.columns]

    result = pd.concat(
        [
            left_matched,
            right_matched,
            match_df[["_score"]].reset_index(drop=True),
        ],
        axis=1,
    )
    return result


def _fuzzy_join_polars(
    left: Any,
    right: Any,
    left_col: str,
    right_col: str,
    scorer: Scorer,
    threshold: float,
    how: str,
    limit: int,
) -> Any:
    import polars as pl

    right_values = right[right_col].cast(pl.Utf8).to_list()
    left_values = left[left_col].cast(pl.Utf8).to_list()

    left_indices: list[int] = []
    right_indices: list[int | None] = []
    scores: list[float | None] = []

    for i, left_val in enumerate(left_values):
        if limit == 1:
            result = reclink.match_best(str(left_val), right_values, scorer, threshold)
            if result is not None:
                _, score, right_idx = result
                left_indices.append(i)
                right_indices.append(right_idx)
                scores.append(score)
            elif how == "left":
                left_indices.append(i)
                right_indices.append(None)
                scores.append(None)
        else:
            results = reclink.match_batch(str(left_val), right_values, scorer, threshold, limit)
            if results:
                for _, score, right_idx in results:
                    left_indices.append(i)
                    right_indices.append(right_idx)
                    scores.append(score)
            elif how == "left":
                left_indices.append(i)
                right_indices.append(None)
                scores.append(None)

    if not left_indices:
        return pl.DataFrame()

    # Build result from indices
    left_matched = left[left_indices]

    right_cols: dict[str, list[Any]] = {f"{col}_right": [] for col in right.columns}
    for ridx in right_indices:
        if ridx is not None:
            row = right.row(ridx)
            for col_name, val in zip(right.columns, row, strict=True):
                right_cols[f"{col_name}_right"].append(val)
        else:
            for col_name in right.columns:
                right_cols[f"{col_name}_right"].append(None)

    right_df = pl.DataFrame(right_cols)
    score_df = pl.DataFrame({"_score": scores})

    return pl.concat([left_matched, right_df, score_df], how="horizontal")
