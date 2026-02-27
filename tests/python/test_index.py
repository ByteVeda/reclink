"""Tests for incremental index updates."""

from __future__ import annotations

import tempfile
from pathlib import Path

from reclink import BkTree, NgramIndex, VpTree


class TestBkTreeIncremental:
    def test_insert_after_build(self) -> None:
        tree = BkTree.build(["hello", "world"], "levenshtein")
        idx = tree.insert("hallo")
        assert idx == 2
        assert len(tree) == 3
        results = tree.find_within("hallo", 0)
        assert any(r[1] == 2 for r in results)

    def test_remove_excludes_from_search(self) -> None:
        tree = BkTree.build(["hello", "hallo", "world"], "levenshtein")
        assert 1 in tree
        assert tree.remove(1)
        assert 1 not in tree
        assert len(tree) == 2
        results = tree.find_within("hallo", 0)
        assert not any(r[1] == 1 for r in results)

    def test_contains(self) -> None:
        tree = BkTree.build(["hello"], "levenshtein")
        assert 0 in tree
        assert 99 not in tree

    def test_save_load_preserves_state(self) -> None:
        tree = BkTree.build(["hello", "world"], "levenshtein")
        tree.insert("hallo")
        tree.remove(1)  # remove "world"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "tree.bin")
            tree.save(path)
            loaded = BkTree.load(path)
            assert len(loaded) == 2
            assert 1 not in loaded


class TestVpTreeIncremental:
    def test_insert_after_build(self) -> None:
        tree = VpTree.build(["hello", "world"], "jaro_winkler")
        idx = tree.insert("hallo")
        assert idx == 2
        assert len(tree) == 3
        results = tree.find_within("hallo", 0.01)
        assert any(r[1] == 2 for r in results)

    def test_remove_excludes_from_search(self) -> None:
        tree = VpTree.build(["hello", "world", "hallo"], "jaro_winkler")
        assert 2 in tree
        assert tree.remove(2)
        assert 2 not in tree
        assert len(tree) == 2
        results = tree.find_within("hallo", 0.01)
        assert not any(r[1] == 2 for r in results)

    def test_rebuild(self) -> None:
        tree = VpTree.build(["hello", "world"], "jaro_winkler")
        tree.insert("hallo")
        tree.remove(1)
        tree.rebuild()
        assert len(tree) == 2
        assert 0 in tree
        assert 1 not in tree
        assert 2 in tree

    def test_contains(self) -> None:
        tree = VpTree.build(["hello"], "jaro_winkler")
        assert 0 in tree
        assert 99 not in tree


class TestNgramIndexIncremental:
    def test_insert_after_build(self) -> None:
        idx = NgramIndex.build(["hello", "world"], 2)
        new_idx = idx.insert("help")
        assert new_idx == 2
        assert len(idx) == 3
        results = idx.search("help", 2)
        indices = [r[1] for r in results]
        assert 2 in indices

    def test_remove_excludes_from_search(self) -> None:
        idx = NgramIndex.build(["hello", "help", "world"], 2)
        assert 1 in idx
        assert idx.remove(1)
        assert 1 not in idx
        assert len(idx) == 2
        results = idx.search("help", 2)
        indices = [r[1] for r in results]
        assert 1 not in indices

    def test_contains(self) -> None:
        idx = NgramIndex.build(["hello"], 2)
        assert 0 in idx
        assert 99 not in idx
