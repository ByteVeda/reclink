"""Pandas Series and DataFrame accessors for reclink.

Registers ``df["col"].reclink.match_best(...)`` and ``df.reclink.fuzzy_merge(...)``.
"""

from __future__ import annotations

import pandas as pd

import reclink


@pd.api.extensions.register_series_accessor("reclink")
class ReclinkSeriesAccessor:
    """Reclink accessor for Pandas Series."""

    def __init__(self, obj: pd.Series) -> None:
        self._obj = obj

    def match_best(
        self,
        candidates: list[str],
        scorer: str = "jaro_winkler",
        threshold: float | None = None,
    ) -> pd.Series:
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
        pd.Series
            Series of (matched_string, score, index) tuples or None.
        """
        return self._obj.map(lambda x: reclink.match_best(str(x), candidates, scorer, threshold))

    def phonetic(self, algorithm: str = "soundex") -> pd.Series:
        """Apply phonetic encoding to each value.

        Parameters
        ----------
        algorithm : str, optional
            Phonetic algorithm name (default "soundex").

        Returns
        -------
        pd.Series
            Series of phonetic codes.
        """
        encoder = getattr(reclink, algorithm)
        return self._obj.map(lambda x: encoder(str(x)))

    def deduplicate(
        self,
        threshold: float = 0.85,
        scorer: str = "jaro_winkler",
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
        values = [str(v) for v in self._obj.tolist()]
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


@pd.api.extensions.register_dataframe_accessor("reclink")
class ReclinkDataFrameAccessor:
    """Reclink accessor for Pandas DataFrames."""

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = obj

    def fuzzy_merge(
        self,
        right: pd.DataFrame,
        left_on: str,
        right_on: str,
        scorer: str = "jaro_winkler",
        threshold: float = 0.8,
    ) -> pd.DataFrame:
        """Fuzzy join two DataFrames on string columns.

        Parameters
        ----------
        right : pd.DataFrame
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
        pd.DataFrame
            Merged DataFrame with best matches and similarity scores.
        """
        right_values = right[right_on].astype(str).tolist()
        matches = []
        for idx, val in zip(self._obj.index, self._obj[left_on].astype(str), strict=True):
            result = reclink.match_best(val, right_values, scorer, threshold)
            if result is not None:
                _, score, right_idx = result
                matches.append(
                    {"_left_idx": idx, "_right_idx": right.index[right_idx], "_score": score}
                )

        if not matches:
            return pd.DataFrame()

        match_df = pd.DataFrame(matches)
        merged = self._obj.loc[match_df["_left_idx"].values].reset_index(drop=True)
        right_matched = right.loc[match_df["_right_idx"].values].reset_index(drop=True)
        merged = pd.concat(
            [
                merged,
                right_matched.add_suffix("_right"),
                match_df[["_score"]].reset_index(drop=True),
            ],
            axis=1,
        )
        return merged
