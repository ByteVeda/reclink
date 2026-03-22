"""Automatic threshold optimization for record linkage classifiers.

Given labeled comparison vectors, finds the optimal classification
threshold that maximizes a chosen criterion (F1, precision, or recall).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ThresholdResult:
    """Result of threshold optimization.

    Attributes
    ----------
    threshold : float
        The optimal threshold.
    f1 : float
        F1 score at the optimal threshold.
    precision : float
        Precision at the optimal threshold.
    recall : float
        Recall at the optimal threshold.
    all_results : list of dict
        Metrics at each evaluated threshold.
    """

    threshold: float
    f1: float
    precision: float
    recall: float
    all_results: list[dict[str, float]] = field(default_factory=list)


def optimize_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    criterion: str = "f1",
    n_thresholds: int = 100,
) -> ThresholdResult:
    """Find the optimal classification threshold for scored pairs.

    Parameters
    ----------
    scores : sequence of float
        Aggregate similarity scores for each pair.
    labels : sequence of bool
        True labels (True = match, False = non-match).
    criterion : str, optional
        Optimization criterion: ``"f1"``, ``"precision"``, or
        ``"recall"`` (default ``"f1"``).
    n_thresholds : int, optional
        Number of thresholds to evaluate (default 100).

    Returns
    -------
    ThresholdResult
        The optimal threshold and associated metrics.

    Raises
    ------
    ValueError
        If inputs are empty or criterion is invalid.

    Examples
    --------
    >>> scores = [0.9, 0.8, 0.3, 0.2]
    >>> labels = [True, True, False, False]
    >>> result = optimize_threshold(scores, labels)
    >>> 0.0 < result.threshold < 1.0
    True
    """
    if len(scores) != len(labels):
        msg = "scores and labels must have the same length"
        raise ValueError(msg)
    if not scores:
        msg = "scores must not be empty"
        raise ValueError(msg)
    if criterion not in ("f1", "precision", "recall"):
        msg = f"criterion must be 'f1', 'precision', or 'recall', got '{criterion}'"
        raise ValueError(msg)

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return ThresholdResult(
            threshold=min_score,
            f1=0.0,
            precision=0.0,
            recall=0.0,
        )

    step = (max_score - min_score) / n_thresholds
    best_threshold = min_score
    best_value = -1.0
    best_metrics: dict[str, float] = {}
    all_results: list[dict[str, float]] = []

    for i in range(n_thresholds + 1):
        threshold = min_score + i * step
        tp = sum(1 for s, lab in zip(scores, labels, strict=True) if s >= threshold and lab)
        fp = sum(1 for s, lab in zip(scores, labels, strict=True) if s >= threshold and not lab)
        fn = sum(1 for s, lab in zip(scores, labels, strict=True) if s < threshold and lab)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        all_results.append(metrics)

        value = metrics[criterion]
        if value > best_value:
            best_value = value
            best_threshold = threshold
            best_metrics = metrics

    return ThresholdResult(
        threshold=best_threshold,
        f1=best_metrics.get("f1", 0.0),
        precision=best_metrics.get("precision", 0.0),
        recall=best_metrics.get("recall", 0.0),
        all_results=all_results,
    )
