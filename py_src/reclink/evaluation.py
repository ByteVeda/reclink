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


def scored_pairs_from_results(results: Any) -> list[tuple[str, str, float]]:
    """Extract scored pairs from pipeline output.

    Parameters
    ----------
    results : list of dict or DataFrame
        Pipeline output with 'left_id', 'right_id', and 'score' columns/keys.

    Returns
    -------
    list of (str, str, float)
        Normalized ``(left_id, right_id, score)`` triples.
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

    out: list[tuple[str, str, float]] = []
    for r in rows:
        a, b = _normalize_pair(str(r["left_id"]), str(r["right_id"]))
        out.append((a, b, float(r["score"])))
    return out


def roc_curve(
    scored_pairs: list[tuple[str, str, float]],
    true_matches: set[tuple[str, str]],
    all_pairs_count: int | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, list[float]]:
    """Compute ROC curve points from scored pairs.

    Parameters
    ----------
    scored_pairs : list of (str, str, float)
        Scored pair triples as returned by :func:`scored_pairs_from_results`.
    true_matches : set of (str, str)
        Ground truth match pairs.
    all_pairs_count : int or None
        Total number of possible pairs (TN + FP denominator).
        Required for FPR computation. If None, FPR is set to 0.0
        at every threshold.
    thresholds : list of float or None
        Thresholds to evaluate. If None, uses unique scores
        (sorted descending) plus a value below the minimum.

    Returns
    -------
    dict
        ``{"fpr": [...], "tpr": [...], "thresholds": [...]}``.
    """
    true_norm = _normalize_pairs(true_matches)

    if thresholds is None:
        scores = sorted({s for _, _, s in scored_pairs}, reverse=True)
        thresholds = [*scores, scores[-1] - 1.0] if scores else [0.0]

    fpr_list: list[float] = []
    tpr_list: list[float] = []

    n_pos = len(true_norm)
    n_neg = (all_pairs_count - n_pos) if all_pairs_count is not None else None

    for t in thresholds:
        predicted = {_normalize_pair(a, b) for a, b, s in scored_pairs if s >= t}
        tp = len(predicted & true_norm)
        fp = len(predicted - true_norm)
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg is not None and n_neg > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return {"fpr": fpr_list, "tpr": tpr_list, "thresholds": thresholds}


def auc(fpr: list[float], tpr: list[float]) -> float:
    """Compute area under the ROC curve using the trapezoidal rule.

    Parameters
    ----------
    fpr : list of float
        False positive rates.
    tpr : list of float
        True positive rates.

    Returns
    -------
    float
        Area under the curve in [0, 1].
    """
    # Sort by FPR ascending
    points = sorted(zip(fpr, tpr, strict=True), key=lambda p: (p[0], p[1]))
    area = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        avg_y = (points[i][1] + points[i - 1][1]) / 2.0
        area += dx * avg_y
    return area


def optimal_threshold(
    scored_pairs: list[tuple[str, str, float]],
    true_matches: set[tuple[str, str]],
    criterion: str = "f1",
) -> dict[str, float]:
    """Find the threshold that maximizes the given criterion.

    Parameters
    ----------
    scored_pairs : list of (str, str, float)
        Scored pair triples.
    true_matches : set of (str, str)
        Ground truth match pairs.
    criterion : str
        One of ``"f1"``, ``"precision"``, or ``"recall"``.

    Returns
    -------
    dict
        ``{"threshold": float, "f1": float, "precision": float,
        "recall": float}``.
    """
    true_norm = _normalize_pairs(true_matches)
    thresholds = sorted({s for _, _, s in scored_pairs})

    best: dict[str, float] = {
        "threshold": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    best_value = -1.0

    for t in thresholds:
        predicted = {_normalize_pair(a, b) for a, b, s in scored_pairs if s >= t}
        p = precision(predicted, true_norm)
        r = recall(predicted, true_norm)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        if criterion == "f1":
            value = f
        elif criterion == "precision":
            value = p
        elif criterion == "recall":
            value = r
        else:
            msg = f"Unknown criterion: {criterion}. Expected: f1, precision, recall"
            raise ValueError(msg)

        if value > best_value:
            best_value = value
            best = {"threshold": t, "f1": f, "precision": p, "recall": r}

    return best
