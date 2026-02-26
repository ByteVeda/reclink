"""Tests for the evaluation module."""

from __future__ import annotations

from reclink.evaluation import confusion_matrix, f1_score, pairs_from_results, precision, recall


class TestPrecision:
    def test_perfect_precision(self) -> None:
        predicted = {("1", "2"), ("3", "4")}
        truth = {("1", "2"), ("3", "4"), ("5", "6")}
        assert precision(predicted, truth) == 1.0

    def test_no_correct(self) -> None:
        predicted = {("1", "2")}
        truth = {("3", "4")}
        assert precision(predicted, truth) == 0.0

    def test_empty_predicted(self) -> None:
        assert precision(set(), {("1", "2")}) == 0.0


class TestRecall:
    def test_perfect_recall(self) -> None:
        predicted = {("1", "2"), ("3", "4")}
        truth = {("1", "2"), ("3", "4")}
        assert recall(predicted, truth) == 1.0

    def test_partial_recall(self) -> None:
        predicted = {("1", "2")}
        truth = {("1", "2"), ("3", "4")}
        assert recall(predicted, truth) == 0.5

    def test_empty_truth(self) -> None:
        assert recall({("1", "2")}, set()) == 0.0


class TestF1Score:
    def test_perfect_f1(self) -> None:
        pairs = {("1", "2")}
        assert f1_score(pairs, pairs) == 1.0

    def test_f1_combined(self) -> None:
        predicted = {("1", "2"), ("3", "4")}
        truth = {("1", "2"), ("5", "6")}
        # precision = 1/2, recall = 1/2, f1 = 2*(0.5*0.5)/(0.5+0.5) = 0.5
        assert abs(f1_score(predicted, truth) - 0.5) < 1e-10

    def test_empty_both(self) -> None:
        assert f1_score(set(), set()) == 0.0


class TestConfusionMatrix:
    def test_basic(self) -> None:
        predicted = {("1", "2"), ("3", "4")}
        truth = {("1", "2"), ("5", "6")}
        cm = confusion_matrix(predicted, truth)
        assert cm["tp"] == 1
        assert cm["fp"] == 1
        assert cm["fn"] == 1
        assert "tn" not in cm

    def test_with_all_pairs(self) -> None:
        predicted = {("1", "2")}
        truth = {("1", "2")}
        all_pairs = {("1", "2"), ("1", "3"), ("2", "3")}
        cm = confusion_matrix(predicted, truth, all_pairs)
        assert cm["tp"] == 1
        assert cm["fp"] == 0
        assert cm["fn"] == 0
        assert cm["tn"] == 2


class TestPairNormalization:
    def test_reversed_pairs_match(self) -> None:
        """(a, b) and (b, a) should be treated as the same pair."""
        predicted = {("2", "1")}
        truth = {("1", "2")}
        assert precision(predicted, truth) == 1.0
        assert recall(predicted, truth) == 1.0


class TestPairsFromResults:
    def test_from_list_of_dicts(self) -> None:
        results = [
            {"left_id": "1", "right_id": "2", "score": 0.9},
            {"left_id": "3", "right_id": "4", "score": 0.8},
        ]
        pairs = pairs_from_results(results)
        assert ("1", "2") in pairs
        assert ("3", "4") in pairs
        assert len(pairs) == 2
