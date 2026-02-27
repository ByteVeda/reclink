"""Polars Series namespace extension for reclink.

Registers ``series.reclink.match_best(...)`` etc.
"""

from __future__ import annotations

import polars as pl

import reclink


class ReclinkNamespace:
    """Reclink namespace for Polars Series."""

    def __init__(self, series: pl.Series) -> None:
        self._series = series

    def match_best(
        self,
        candidates: list[str],
        scorer: str = "jaro_winkler",
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


# Register namespace on pl.Series
pl.api.register_series_namespace("reclink")(ReclinkNamespace)
