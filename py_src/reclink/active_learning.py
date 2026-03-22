"""Active learning for record linkage.

Provides an active learning loop that identifies uncertain pairs near the
classification boundary and uses human labels to optimize thresholds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reclink._core import Scorer


class ActiveLearner:
    """Active learning wrapper for threshold-based record linkage.

    Identifies pairs near the classification boundary and uses manual
    labels to find the optimal threshold.

    Parameters
    ----------
    scorer : str, optional
        Metric name (default ``"jaro_winkler"``).
    threshold : float, optional
        Initial classification threshold (default 0.85).

    Examples
    --------
    >>> learner = ActiveLearner(scorer="jaro_winkler", threshold=0.85)
    >>> pairs = learner.uncertain_pairs(records, n=10)
    >>> # Label the pairs manually...
    >>> labels = [{"left": 0, "right": 1, "label": "match"}, ...]
    >>> learner.update_from_labels(labels)
    >>> print(f"New threshold: {learner.threshold}")
    """

    def __init__(
        self,
        scorer: Scorer = "jaro_winkler",
        threshold: float = 0.85,
    ) -> None:
        self.scorer: Scorer = scorer
        self.threshold = threshold
        self._scores: list[dict[str, Any]] = []
        self._labels: list[dict[str, Any]] = []

    def score_pairs(
        self,
        records: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Score all pairs and store results for active learning.

        Parameters
        ----------
        records : sequence of str
            List of strings to compare pairwise.

        Returns
        -------
        list of dict
            Each dict has ``left``, ``right``, ``score`` keys.
        """
        import reclink

        self._scores = []
        n = len(records)
        for i in range(n):
            for j in range(i + 1, n):
                score = reclink.cdist([records[i]], [records[j]], self.scorer)[0, 0]
                self._scores.append({"left": i, "right": j, "score": float(score)})
        return self._scores

    def uncertain_pairs(
        self,
        records: Sequence[str],
        n: int = 10,
    ) -> list[dict[str, Any]]:
        """Find the most uncertain pairs near the threshold.

        Parameters
        ----------
        records : sequence of str
            List of strings to compare.
        n : int, optional
            Number of uncertain pairs to return (default 10).

        Returns
        -------
        list of dict
            Pairs sorted by closeness to threshold. Each dict has
            ``left`` (index), ``right`` (index), ``left_value``,
            ``right_value``, ``score``.
        """
        if not self._scores:
            self.score_pairs(records)

        # Sort by distance from threshold
        sorted_pairs = sorted(
            self._scores,
            key=lambda p: abs(p["score"] - self.threshold),
        )

        result = []
        for pair in sorted_pairs[:n]:
            result.append(
                {
                    "left": pair["left"],
                    "right": pair["right"],
                    "left_value": records[pair["left"]],
                    "right_value": records[pair["right"]],
                    "score": pair["score"],
                }
            )
        return result

    def update_from_labels(
        self,
        labels: list[dict[str, Any]],
    ) -> float:
        """Update the threshold based on manual labels.

        Finds the optimal threshold that maximizes separation between
        labeled matches and non-matches.

        Parameters
        ----------
        labels : list of dict
            Each dict must have ``left`` (int), ``right`` (int), and
            ``label`` (``"match"`` or ``"non_match"``).

        Returns
        -------
        float
            The updated threshold.
        """
        self._labels.extend(labels)

        match_scores: list[float] = []
        non_match_scores: list[float] = []

        score_lookup = {(p["left"], p["right"]): p["score"] for p in self._scores}

        for label in self._labels:
            key = (label["left"], label["right"])
            score = score_lookup.get(key)
            if score is None:
                continue
            if label["label"] == "match":
                match_scores.append(score)
            else:
                non_match_scores.append(score)

        if match_scores and non_match_scores:
            # Optimal threshold: midpoint between highest non-match
            # and lowest match
            max_non_match = max(non_match_scores)
            min_match = min(match_scores)
            self.threshold = (max_non_match + min_match) / 2.0
        elif match_scores:
            self.threshold = min(match_scores) - 0.01
        elif non_match_scores:
            self.threshold = max(non_match_scores) + 0.01

        return self.threshold

    def classify(
        self,
        records: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Classify all pairs using the current threshold.

        Parameters
        ----------
        records : sequence of str
            List of strings to compare.

        Returns
        -------
        list of dict
            Pairs classified as matches (score >= threshold).
        """
        if not self._scores:
            self.score_pairs(records)

        return [p for p in self._scores if p["score"] >= self.threshold]
