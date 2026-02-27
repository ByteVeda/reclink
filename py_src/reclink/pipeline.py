"""High-level record linkage pipeline with DataFrame support.

Wraps the Rust pipeline with a Pythonic builder API and optional
pandas/polars integration.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from reclink._core import PyPipeline, PyRecord

if TYPE_CHECKING:
    from collections.abc import Sequence


class ReclinkPipeline:
    """Builder-pattern pipeline for record linkage and deduplication.

    Examples
    --------
    >>> pipeline = (
    ...     ReclinkPipeline.builder()
    ...     .preprocess_lowercase(["first_name", "last_name"])
    ...     .block_phonetic("last_name", algorithm="soundex")
    ...     .compare_string("first_name", metric="jaro_winkler")
    ...     .compare_string("last_name", metric="jaro_winkler")
    ...     .classify_threshold(0.85)
    ...     .build()
    ... )
    """

    def __init__(self, inner: PyPipeline) -> None:
        self._inner = inner

    @staticmethod
    def builder() -> PipelineBuilder:
        """Create a new pipeline builder."""
        return PipelineBuilder()

    def dedup(
        self,
        data: Any,
        id_column: str = "id",
    ) -> Any:
        """Deduplicate a dataset.

        Parameters
        ----------
        data : DataFrame or list of dicts
            Input data. Accepts pandas DataFrame, polars DataFrame,
            or a list of dicts with string values.
        id_column : str
            Column name for record identifiers.

        Returns
        -------
        DataFrame or list of dict
            Match results with left_id, right_id, score, and scores.
            Returns the same container type as the input.
        """
        records = _to_records(data, id_column)
        results = self._inner.dedup(records)
        rows = [
            {
                "left_id": r.left_id,
                "right_id": r.right_id,
                "score": r.score,
                "scores": r.scores,
            }
            for r in results
        ]
        return _convert_output(rows, data)

    def dedup_cluster(
        self,
        data: Any,
        id_column: str = "id",
    ) -> Any:
        """Deduplicate and cluster a dataset.

        Parameters
        ----------
        data : DataFrame or list of dicts
            Input data.
        id_column : str
            Column name for record identifiers.

        Returns
        -------
        DataFrame or list of list of str
            Groups of matching record IDs. When input is a DataFrame,
            returns a DataFrame with cluster_id and record_id columns.
        """
        records = _to_records(data, id_column)
        clusters = self._inner.dedup_cluster(records)
        return _clusters_to_output(clusters, data)

    def link(
        self,
        left: Any,
        right: Any,
        id_column: str = "id",
    ) -> Any:
        """Link two datasets.

        Parameters
        ----------
        left : DataFrame or list of dicts
            First dataset.
        right : DataFrame or list of dicts
            Second dataset.
        id_column : str
            Column name for record identifiers.

        Returns
        -------
        DataFrame or list of dict
            Match results with left_id, right_id, score, and scores.
            Returns the same container type as the left input.
        """
        left_records = _to_records(left, id_column)
        right_records = _to_records(right, id_column)
        results = self._inner.link(left_records, right_records)
        rows = [
            {
                "left_id": r.left_id,
                "right_id": r.right_id,
                "score": r.score,
                "scores": r.scores,
            }
            for r in results
        ]
        return _convert_output(rows, left)


class PipelineBuilder:
    """Fluent builder for constructing a ReclinkPipeline."""

    def __init__(self) -> None:
        self._inner = PyPipeline()

    def preprocess(self, field: str, operations: Sequence[str]) -> PipelineBuilder:
        """Apply preprocessing operations to a field before comparison.

        Parameters
        ----------
        field : str
            Field name to preprocess.
        operations : sequence of str
            Operation names: "fold_case", "normalize_whitespace",
            "strip_punctuation", "standardize_name",
            "normalize_unicode_nfc", etc.
        """
        self._inner.preprocess(field, list(operations))
        return self

    def preprocess_lowercase(self, fields: Sequence[str]) -> PipelineBuilder:
        """Lowercase the specified fields before comparison.

        .. deprecated::
            Use :meth:`preprocess` with ``["fold_case"]`` instead.

        Parameters
        ----------
        fields : sequence of str
            Field names to lowercase.
        """
        warnings.warn(
            "preprocess_lowercase is deprecated, use .preprocess(field, ['fold_case'])",
            DeprecationWarning,
            stacklevel=2,
        )
        for field in fields:
            self._inner.preprocess(field, ["fold_case"])
        return self

    def block_exact(self, field: str) -> PipelineBuilder:
        """Block on exact match of a field.

        Parameters
        ----------
        field : str
            Field name for exact blocking.
        """
        self._inner.block_exact(field)
        return self

    def block_phonetic(self, field: str, algorithm: str = "soundex") -> PipelineBuilder:
        """Block on phonetic encoding of a field.

        Parameters
        ----------
        field : str
            Field name for phonetic blocking.
        algorithm : str
            One of "soundex", "metaphone", "double_metaphone", "nysiis".
        """
        self._inner.block_phonetic(field, algorithm)
        return self

    def block_sorted_neighborhood(self, field: str, window: int = 3) -> PipelineBuilder:
        """Block using sorted neighborhood method.

        Parameters
        ----------
        field : str
            Field name to sort on.
        window : int
            Window size for neighbor comparison.
        """
        self._inner.block_sorted_neighborhood(field, window)
        return self

    def block_qgram(self, field: str, q: int = 3, threshold: int = 1) -> PipelineBuilder:
        """Block using q-gram (n-gram) overlap.

        Parameters
        ----------
        field : str
            Field name for q-gram blocking.
        q : int
            N-gram size.
        threshold : int
            Minimum shared q-grams to form a candidate pair.
        """
        self._inner.block_qgram(field, q, threshold)
        return self

    def compare_string(self, field: str, metric: str = "jaro_winkler") -> PipelineBuilder:
        """Compare a text field using a string similarity metric.

        Parameters
        ----------
        field : str
            Field name to compare.
        metric : str
            Metric name (e.g., "jaro_winkler", "levenshtein").
        """
        self._inner.compare_string(field, metric)
        return self

    def compare_exact(self, field: str) -> PipelineBuilder:
        """Compare a field for exact equality.

        Parameters
        ----------
        field : str
            Field name to compare.
        """
        self._inner.compare_exact(field)
        return self

    def compare_numeric(self, field: str, max_diff: float = 10.0) -> PipelineBuilder:
        """Compare a numeric field.

        Parameters
        ----------
        field : str
            Field name to compare.
        max_diff : float
            Maximum difference for zero similarity.
        """
        self._inner.compare_numeric(field, max_diff)
        return self

    def compare_date(self, field: str) -> PipelineBuilder:
        """Compare a date field.

        Parameters
        ----------
        field : str
            Field name to compare.
        """
        self._inner.compare_date(field)
        return self

    def classify_threshold(self, threshold: float) -> PipelineBuilder:
        """Classify using average score threshold.

        Parameters
        ----------
        threshold : float
            Score threshold for match classification.
        """
        self._inner.classify_threshold(threshold)
        return self

    def classify_weighted(self, weights: Sequence[float], threshold: float) -> PipelineBuilder:
        """Classify using weighted sum of scores.

        Parameters
        ----------
        weights : sequence of float
            Per-field weights.
        threshold : float
            Weighted sum threshold for match classification.
        """
        self._inner.classify_weighted(list(weights), threshold)
        return self

    def compare_phonetic(self, field: str, algorithm: str = "soundex") -> PipelineBuilder:
        """Compare a field using phonetic encoding (binary: match → 1.0, else → 0.0).

        Parameters
        ----------
        field : str
            Field name to compare.
        algorithm : str
            One of "soundex", "metaphone", "double_metaphone", "nysiis".
        """
        self._inner.compare_phonetic(field, algorithm)
        return self

    def classify_fellegi_sunter(
        self,
        m_probs: Sequence[float],
        u_probs: Sequence[float],
        upper: float,
        lower: float,
    ) -> PipelineBuilder:
        """Classify using Fellegi-Sunter probabilistic model.

        Parameters
        ----------
        m_probs : sequence of float
            P(agree | match) for each field.
        u_probs : sequence of float
            P(agree | non-match) for each field.
        upper : float
            Upper threshold for definite matches.
        lower : float
            Lower threshold for definite non-matches.
        """
        self._inner.classify_fellegi_sunter(list(m_probs), list(u_probs), upper, lower)
        return self

    def classify_fellegi_sunter_auto(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        initial_p_match: float = 0.1,
    ) -> PipelineBuilder:
        """Classify using Fellegi-Sunter with EM-estimated parameters.

        Parameters
        ----------
        max_iterations : int
            Maximum EM iterations.
        convergence_threshold : float
            Convergence threshold for parameter changes.
        initial_p_match : float
            Initial prior probability of a match.
        """
        self._inner.classify_fellegi_sunter_auto(
            max_iterations, convergence_threshold, initial_p_match
        )
        return self

    def cluster_connected_components(self) -> PipelineBuilder:
        """Enable connected-component clustering of results."""
        self._inner.cluster_connected_components()
        return self

    def cluster_hierarchical(
        self, linkage: str = "single", threshold: float = 0.5
    ) -> PipelineBuilder:
        """Enable hierarchical agglomerative clustering of results.

        Parameters
        ----------
        linkage : str
            Linkage criterion: "single", "complete", or "average".
        threshold : float
            Distance threshold for merging clusters.
        """
        self._inner.cluster_hierarchical(linkage, threshold)
        return self

    def block_lsh(self, field: str, num_hashes: int = 100, num_bands: int = 20) -> PipelineBuilder:
        """Block using Locality-Sensitive Hashing (MinHash + banding).

        Parameters
        ----------
        field : str
            Field name for LSH blocking.
        num_hashes : int
            Number of hash functions (signature length).
        num_bands : int
            Number of bands for the banding technique.
        """
        self._inner.block_lsh(field, num_hashes, num_bands)
        return self

    def block_canopy(
        self,
        field: str,
        t_tight: float = 0.9,
        t_loose: float = 0.5,
        metric: str = "jaro_winkler",
    ) -> PipelineBuilder:
        """Block using canopy clustering with two thresholds.

        Parameters
        ----------
        field : str
            Field name for canopy blocking.
        t_tight : float
            Tight threshold — records within this similarity are strongly linked.
        t_loose : float
            Loose threshold — records within this similarity are candidates.
        metric : str
            Similarity metric name (e.g., "jaro_winkler", "levenshtein").
        """
        self._inner.block_canopy(field, t_tight, t_loose, metric)
        return self

    def block_numeric(self, field: str, bucket_size: float = 5.0) -> PipelineBuilder:
        """Block using numeric bucket ranges.

        Parameters
        ----------
        field : str
            Field name for numeric blocking.
        bucket_size : float
            Width of each bucket (e.g., 5.0 for age ranges 20-24, 25-29).
        """
        self._inner.block_numeric(field, bucket_size)
        return self

    def block_date(self, field: str, resolution: str = "year") -> PipelineBuilder:
        """Block by truncating a date field to the given resolution.

        Parameters
        ----------
        field : str
            Field name for date blocking.
        resolution : str
            One of "year", "month", "day".
        """
        self._inner.block_date(field, resolution)
        return self

    def build(self) -> ReclinkPipeline:
        """Build the configured pipeline."""
        return ReclinkPipeline(self._inner)


def _to_records(data: Any, id_column: str) -> list[PyRecord]:
    """Convert input data to a list of PyRecord objects."""
    rows: list[dict[str, str]] = []

    # Pandas DataFrame
    if hasattr(data, "iterrows"):
        for _, row in data.iterrows():
            rows.append({str(k): str(v) for k, v in row.items()})
    # Polars DataFrame
    elif hasattr(data, "to_dicts"):
        for row_dict in data.to_dicts():
            rows.append({str(k): str(v) for k, v in row_dict.items()})
    # List of dicts
    elif isinstance(data, list):
        rows = [{str(k): str(v) for k, v in d.items()} for d in data]
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            "Expected pandas DataFrame, polars DataFrame, or list of dicts."
        )

    records: list[PyRecord] = []
    for row in rows:
        record_id = row.pop(id_column, str(len(records)))
        rec = PyRecord(record_id)
        for k, v in row.items():
            rec.set_field(k, v)
        records.append(rec)

    return records


def _convert_output(results: list[dict[str, Any]], original_input: Any) -> Any:
    """Return results as the same container type as the input."""
    if hasattr(original_input, "iterrows"):  # pandas DataFrame
        import pandas as pd

        return pd.DataFrame(results)
    if hasattr(original_input, "to_dicts"):  # polars DataFrame
        import polars as pl

        return pl.DataFrame(results)
    return results


def _clusters_to_output(clusters: list[list[str]], original_input: Any) -> Any:
    """Return cluster results as the same container type as the input."""
    if hasattr(original_input, "iterrows"):  # pandas DataFrame
        import pandas as pd

        rows = [
            {"cluster_id": i, "record_id": rid} for i, group in enumerate(clusters) for rid in group
        ]
        return pd.DataFrame(rows)
    if hasattr(original_input, "to_dicts"):  # polars DataFrame
        import polars as pl

        rows = [
            {"cluster_id": i, "record_id": rid} for i, group in enumerate(clusters) for rid in group
        ]
        return pl.DataFrame(rows)
    return clusters
