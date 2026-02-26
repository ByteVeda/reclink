"""Evaluation metrics for measuring record linkage quality.

Provides precision, recall, F1 score, and confusion matrix computation
for comparing predicted matches against ground truth.
"""

from __future__ import annotations

from typing import Any


def _normalize_pair(a: str, b: str) -> tuple[str, str]:
    """Normalize a pair so that (a, b) == (b, a)."""
    return (min(a, b), max(a, b))


def _normalize_pairs(pairs: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Normalize all pairs so order doesn't matter."""
    return {_normalize_pair(a, b) for a, b in pairs}


def precision(predicted: set[tuple[str, str]], true_matches: set[tuple[str, str]]) -> float:
    """Fraction of predicted matches that are correct.

    Parameters
    ----------
    predicted : set of (str, str)
        Predicted match pairs as (id_a, id_b).
    true_matches : set of (str, str)
        Ground truth match pairs as (id_a, id_b).

    Returns
    -------
    float
        Precision score in [0, 1].
    """
    if not predicted:
        return 0.0
    pred_norm = _normalize_pairs(predicted)
    true_norm = _normalize_pairs(true_matches)
    tp = len(pred_norm & true_norm)
    return tp / len(pred_norm)


def recall(predicted: set[tuple[str, str]], true_matches: set[tuple[str, str]]) -> float:
    """Fraction of true matches that were found.

    Parameters
    ----------
    predicted : set of (str, str)
        Predicted match pairs as (id_a, id_b).
    true_matches : set of (str, str)
        Ground truth match pairs as (id_a, id_b).

    Returns
    -------
    float
        Recall score in [0, 1].
    """
    if not true_matches:
        return 0.0
    pred_norm = _normalize_pairs(predicted)
    true_norm = _normalize_pairs(true_matches)
    tp = len(pred_norm & true_norm)
    return tp / len(true_norm)


def f1_score(predicted: set[tuple[str, str]], true_matches: set[tuple[str, str]]) -> float:
    """Harmonic mean of precision and recall.

    Parameters
    ----------
    predicted : set of (str, str)
        Predicted match pairs as (id_a, id_b).
    true_matches : set of (str, str)
        Ground truth match pairs as (id_a, id_b).

    Returns
    -------
    float
        F1 score in [0, 1].
    """
    p = precision(predicted, true_matches)
    r = recall(predicted, true_matches)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def confusion_matrix(
    predicted: set[tuple[str, str]],
    true_matches: set[tuple[str, str]],
    all_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, int]:
    """Compute confusion matrix counts.

    Parameters
    ----------
    predicted : set of (str, str)
        Predicted match pairs.
    true_matches : set of (str, str)
        Ground truth match pairs.
    all_pairs : set of (str, str) or None
        Universe of all possible pairs. Required for TN computation.
        If None, TN is not included in the result.

    Returns
    -------
    dict
        Dictionary with keys 'tp', 'fp', 'fn', and optionally 'tn'.
    """
    pred_norm = _normalize_pairs(predicted)
    true_norm = _normalize_pairs(true_matches)

    tp = len(pred_norm & true_norm)
    fp = len(pred_norm - true_norm)
    fn_ = len(true_norm - pred_norm)

    result: dict[str, int] = {"tp": tp, "fp": fp, "fn": fn_}
    if all_pairs is not None:
        all_norm = _normalize_pairs(all_pairs)
        tn = len(all_norm - pred_norm - true_norm)
        result["tn"] = tn
    return result


def pairs_from_results(results: Any) -> set[tuple[str, str]]:
    """Convert pipeline output to a set of normalized ID pairs.

    Parameters
    ----------
    results : list of dict or DataFrame
        Pipeline output with 'left_id' and 'right_id' columns/keys.

    Returns
    -------
    set of (str, str)
        Normalized pairs.
    """
    rows: list[dict[str, Any]]
    if hasattr(results, "to_dicts"):
        rows = results.to_dicts()
    elif hasattr(results, "to_dict"):
        rows = results.to_dict("records")
    elif isinstance(results, list):
        rows = results
    else:
        raise TypeError(f"Unsupported results type: {type(results)}")

    return {_normalize_pair(str(r["left_id"]), str(r["right_id"])) for r in rows}
