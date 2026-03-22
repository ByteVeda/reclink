"""Tests for Tier 2 features: index structures, clustering, ML classifiers."""

import reclink
from reclink.threshold_optimizer import optimize_threshold


class TestBloomFilter:
    def test_basic(self) -> None:
        bf = reclink.BloomFilter(expected_items=100, false_positive_rate=0.01)
        bf.insert("hello")
        bf.insert("world")
        assert bf.contains("hello")
        assert bf.contains("world")
        assert len(bf) == 2

    def test_no_false_negatives(self) -> None:
        bf = reclink.BloomFilter(expected_items=100)
        items = [f"item_{i}" for i in range(50)]
        for item in items:
            bf.insert(item)
        for item in items:
            assert item in bf

    def test_empty(self) -> None:
        bf = reclink.BloomFilter()
        assert not bf.contains("anything")
        assert len(bf) == 0

    def test_memory_usage(self) -> None:
        bf = reclink.BloomFilter(expected_items=1000)
        assert isinstance(bf.memory_usage(), str)


class TestInvertedIndex:
    def test_whitespace(self) -> None:
        idx = reclink.InvertedIndex.build(
            ["hello world", "world peace", "hello peace"], "whitespace"
        )
        results = idx.search("hello world", 1)
        assert len(results) > 0

    def test_ngram(self) -> None:
        idx = reclink.InvertedIndex.build(["hello", "hallo", "world"], "ngram:2")
        results = idx.search("hello", 1)
        assert any(r[1] == 0 for r in results)

    def test_top_k(self) -> None:
        idx = reclink.InvertedIndex.build(["a b c", "a b", "a", "x y z"])
        results = idx.search_top_k("a b c", 2)
        assert len(results) <= 2

    def test_empty(self) -> None:
        idx = reclink.InvertedIndex.build([])
        assert len(idx) == 0

    def test_vocab_size(self) -> None:
        idx = reclink.InvertedIndex.build(["hello world", "hello"])
        assert idx.vocab_size() == 2


class TestDBSCAN:
    def test_two_clusters(self) -> None:
        sims = [
            (0, 1, 0.9),
            (1, 2, 0.9),
            (0, 2, 0.85),
            (3, 4, 0.9),
            (4, 5, 0.9),
            (3, 5, 0.85),
        ]
        clusters, noise, _labels = reclink.dbscan_cluster(6, sims, 0.8, 2)
        assert len(clusters) == 2
        assert len(noise) == 0

    def test_noise_points(self) -> None:
        sims = [(0, 1, 0.9), (1, 2, 0.9)]
        _clusters, noise, _labels = reclink.dbscan_cluster(4, sims, 0.8, 2)
        assert len(noise) > 0

    def test_empty(self) -> None:
        clusters, _noise, _labels = reclink.dbscan_cluster(0, [], 0.8, 2)
        assert len(clusters) == 0


class TestOPTICS:
    def test_basic(self) -> None:
        sims = [
            (0, 1, 0.95),
            (1, 2, 0.95),
            (0, 2, 0.9),
            (3, 4, 0.95),
            (4, 5, 0.95),
            (3, 5, 0.9),
        ]
        clusters, _noise = reclink.optics_cluster(6, sims, 2, 0.8)
        assert isinstance(clusters, list)

    def test_empty(self) -> None:
        clusters, _noise = reclink.optics_cluster(0, [], 2, 0.8)
        assert len(clusters) == 0


class TestClusterQuality:
    def test_silhouette(self) -> None:
        sims = [
            (0, 1, 0.95),
            (2, 3, 0.95),
            (0, 2, 0.1),
            (0, 3, 0.1),
            (1, 2, 0.1),
            (1, 3, 0.1),
        ]
        labels = [0, 0, 1, 1]
        score = reclink.silhouette_score(4, sims, labels)
        assert score > 0.0

    def test_davies_bouldin(self) -> None:
        sims = [
            (0, 1, 0.95),
            (2, 3, 0.95),
            (0, 2, 0.1),
            (0, 3, 0.1),
            (1, 2, 0.1),
            (1, 3, 0.1),
        ]
        labels = [0, 0, 1, 1]
        db = reclink.davies_bouldin_index(4, sims, labels)
        assert db < 1.0


class TestLogisticRegression:
    def test_train_and_use(self) -> None:
        vectors = [[0.9, 0.95], [0.85, 0.9], [0.1, 0.15], [0.2, 0.1]]
        labels = [True, True, False, False]
        weights, bias, threshold = reclink.train_logistic_regression(vectors, labels)
        assert len(weights) == 2
        assert isinstance(bias, float)
        assert isinstance(threshold, float)


class TestDecisionTree:
    def test_train_and_use(self) -> None:
        vectors = [
            [0.9, 0.95],
            [0.85, 0.9],
            [0.8, 0.85],
            [0.1, 0.15],
            [0.2, 0.1],
            [0.15, 0.2],
        ]
        labels = [True, True, True, False, False, False]
        tree_json = reclink.train_decision_tree(
            vectors, labels, max_depth=3, min_samples_leaf=1, min_samples_split=2
        )
        assert isinstance(tree_json, str)
        assert "feature_index" in tree_json or "prediction" in tree_json


class TestThresholdOptimizer:
    def test_basic(self) -> None:
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        labels = [True, True, True, False, False, False]
        result = optimize_threshold(scores, labels)
        assert 0.0 < result.threshold < 1.0
        assert result.f1 > 0.0

    def test_criterion(self) -> None:
        scores = [0.9, 0.8, 0.3, 0.2]
        labels = [True, True, False, False]
        result = optimize_threshold(scores, labels, criterion="precision")
        assert result.precision > 0.0

    def test_all_results(self) -> None:
        scores = [0.9, 0.1]
        labels = [True, False]
        result = optimize_threshold(scores, labels, n_thresholds=10)
        assert len(result.all_results) == 11  # n_thresholds + 1
