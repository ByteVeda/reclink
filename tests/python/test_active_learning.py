"""Tests for active learning."""

from reclink.active_learning import ActiveLearner


class TestActiveLearner:
    def test_uncertain_pairs(self) -> None:
        learner = ActiveLearner(scorer="jaro_winkler", threshold=0.85)
        records = ["John Smith", "Jon Smith", "Jane Doe", "Alice Johnson"]
        pairs = learner.uncertain_pairs(records, n=3)
        assert len(pairs) <= 3
        assert all("score" in p for p in pairs)
        assert all("left_value" in p for p in pairs)

    def test_update_threshold(self) -> None:
        learner = ActiveLearner(scorer="jaro_winkler", threshold=0.85)
        records = ["John Smith", "Jon Smith", "Jane Doe"]
        learner.score_pairs(records)

        labels = [
            {"left": 0, "right": 1, "label": "match"},
            {"left": 0, "right": 2, "label": "non_match"},
        ]
        new_threshold = learner.update_from_labels(labels)
        assert 0.0 < new_threshold < 1.0

    def test_classify(self) -> None:
        learner = ActiveLearner(scorer="jaro_winkler", threshold=0.5)
        records = ["hello", "hallo", "world"]
        matches = learner.classify(records)
        assert isinstance(matches, list)

    def test_score_pairs(self) -> None:
        learner = ActiveLearner(scorer="jaro_winkler")
        records = ["abc", "abd", "xyz"]
        scores = learner.score_pairs(records)
        # 3 records -> 3 pairs
        assert len(scores) == 3
        assert all("left" in s and "right" in s and "score" in s for s in scores)
