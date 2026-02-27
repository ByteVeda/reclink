"""Polars Series and DataFrame namespace extensions for reclink.

Registers ``series.reclink.match_best(...)`` etc. and
``df.reclink.fuzzy_merge(...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import reclink

if TYPE_CHECKING:
    from reclink._core import Scorer


class ReclinkNamespace:
    """Reclink namespace for Polars Series."""

    def __init__(self, series: pl.Series) -> None:
        self._series = series

    def match_best(
        self,
        candidates: list[str],
        scorer: Scorer = "jaro_winkler",
        threshold: float | None = None,
    ) -> pl.Series:
        """Find the best match for each value in the Series.

        Parameters
        ----------
        candidates : list of str
            Candidate strings to match against.
        scorer : str, optional
            Metric name (default "jaro_winkler").
        threshold : float or None, optional
            Minimum similarity to return a match.

        Returns
        -------
        pl.Series
            Series of best match strings (or None).
        """
        values = self._series.to_list()
        results = []
        for v in values:
            result = reclink.match_best(str(v), candidates, scorer, threshold)
            results.append(result[0] if result is not None else None)
        return pl.Series(self._series.name, results)

    def phonetic(self, algorithm: str = "soundex") -> pl.Series:
        """Apply phonetic encoding to each value.

        Parameters
        ----------
        algorithm : str, optional
            Phonetic algorithm name (default "soundex").

        Returns
        -------
        pl.Series
            Series of phonetic codes.
        """
        encoder = getattr(reclink, algorithm)
        values = self._series.to_list()
        return pl.Series(self._series.name, [encoder(str(v)) for v in values])

    def deduplicate(
        self,
        threshold: float = 0.85,
        scorer: Scorer = "jaro_winkler",
    ) -> list[list[int]]:
        """Find duplicate groups within the Series by index position.

        Parameters
        ----------
        threshold : float, optional
            Minimum similarity to consider a match (default 0.85).
        scorer : str, optional
            Metric name (default "jaro_winkler").

        Returns
        -------
        list of list of int
            Groups of row indices that are duplicates.
        """
        values = [str(v) for v in self._series.to_list()]
        n = len(values)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            results = reclink.match_batch(values[i], values[i + 1 :], scorer, threshold)
            for _, _, j in results:
                union(i, i + 1 + j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        return [g for g in groups.values() if len(g) > 1]


class ReclinkDataFrameNamespace:
    """Reclink namespace for Polars DataFrames."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def fuzzy_merge(
        self,
        right: pl.DataFrame,
        left_on: str,
        right_on: str,
        scorer: Scorer = "jaro_winkler",
        threshold: float = 0.8,
    ) -> pl.DataFrame:
        """Fuzzy join two DataFrames on string columns.

        Parameters
        ----------
        right : pl.DataFrame
            Right DataFrame to merge with.
        left_on : str
            Column name in the left DataFrame.
        right_on : str
            Column name in the right DataFrame.
        scorer : str, optional
            Metric name (default "jaro_winkler").
        threshold : float, optional
            Minimum similarity to include a match (default 0.8).

        Returns
        -------
        pl.DataFrame
            Merged DataFrame with best matches and similarity scores.
        """
        right_values = right[right_on].cast(pl.Utf8).to_list()
        left_indices: list[int] = []
        right_indices: list[int] = []
        scores: list[float] = []

        for i, val in enumerate(self._df[left_on].cast(pl.Utf8).to_list()):
            result = reclink.match_best(str(val), right_values, scorer, threshold)
            if result is not None:
                _, score, right_idx = result
                left_indices.append(i)
                right_indices.append(right_idx)
                scores.append(score)

        if not left_indices:
            return pl.DataFrame()

        left_matched = self._df[left_indices]
        right_matched = right[right_indices].rename({c: f"{c}_right" for c in right.columns})
        score_col = pl.Series("_score", scores)
        return pl.concat([left_matched, right_matched, score_col.to_frame()], how="horizontal")


# Register namespaces
pl.api.register_series_namespace("reclink")(ReclinkNamespace)
pl.api.register_dataframe_namespace("reclink")(ReclinkDataFrameNamespace)
