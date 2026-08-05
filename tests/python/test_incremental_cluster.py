"""Tests for incremental clustering."""

import reclink


class TestIncrementalCluster:
    def test_basic(self) -> None:
        cluster = reclink.IncrementalCluster(metric="jaro_winkler", threshold=0.85)
        cluster_id, is_new, sim = cluster.add_record("John Smith")
        assert cluster_id == 0
        assert is_new is True
        assert sim is None

    def test_same_record_joins_cluster(self) -> None:
        cluster = reclink.IncrementalCluster(metric="jaro_winkler", threshold=0.85)
        cluster.add_record("John Smith")
        cluster_id, is_new, sim = cluster.add_record("John Smith")
        assert cluster_id == 0
        assert is_new is False
        assert sim is not None
        assert sim >= 0.85

    def test_different_records(self) -> None:
        cluster = reclink.IncrementalCluster(metric="jaro_winkler", threshold=0.85)
        cluster.add_record("John Smith")
        cluster.add_record("Alice Johnson")
        assert cluster.cluster_count() == 2

    def test_similar_records(self) -> None:
        cluster = reclink.IncrementalCluster(metric="jaro_winkler", threshold=0.80)
        cluster.add_record("John Smith")
        cluster_id, is_new, _ = cluster.add_record("Jon Smith")
        assert cluster_id == 0
        assert is_new is False

    def test_get_clusters(self) -> None:
        cluster = reclink.IncrementalCluster(metric="jaro_winkler", threshold=0.85)
        cluster.add_record("John Smith")
        cluster.add_record("John Smith")
        cluster.add_record("Alice Johnson")
        clusters = cluster.get_clusters()
        assert len(clusters) == 2
        assert clusters[0] == [0, 1]
        assert clusters[1] == [2]

    def test_record_count(self) -> None:
        cluster = reclink.IncrementalCluster()
        cluster.add_record("a")
        cluster.add_record("b")
        cluster.add_record("c")
        assert cluster.record_count() == 3

    def test_default_params(self) -> None:
        cluster = reclink.IncrementalCluster()
        assert cluster.cluster_count() == 0
        assert cluster.record_count() == 0
