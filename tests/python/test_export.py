"""Tests for the export module."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from reclink.export import (
    export_clusters_csv,
    export_clusters_json,
    export_matches_csv,
    export_matches_json,
)


@pytest.fixture
def match_results() -> list[dict[str, object]]:
    return [
        {"left_id": "1", "right_id": "2", "score": 0.95, "scores": [0.9, 1.0]},
        {"left_id": "3", "right_id": "4", "score": 0.80, "scores": [0.7, 0.9]},
    ]


@pytest.fixture
def clusters() -> list[list[str]]:
    return [["1", "2"], ["3", "4", "5"]]


class TestMatchesCSV:
    def test_export_matches_csv(
        self, match_results: list[dict[str, object]], tmp_path: Path
    ) -> None:
        path = tmp_path / "matches.csv"
        export_matches_csv(match_results, path)

        with path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["left_id", "right_id", "score", "scores"]
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0][0] == "1"
            assert rows[0][1] == "2"


class TestMatchesJSON:
    def test_export_matches_json(
        self, match_results: list[dict[str, object]], tmp_path: Path
    ) -> None:
        path = tmp_path / "matches.json"
        export_matches_json(match_results, path)

        with path.open() as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["left_id"] == "1"
        assert data[0]["right_id"] == "2"


class TestClustersCSV:
    def test_export_clusters_csv(self, clusters: list[list[str]], tmp_path: Path) -> None:
        path = tmp_path / "clusters.csv"
        export_clusters_csv(clusters, path)

        with path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["cluster_id", "record_id"]
            rows = list(reader)
            # 2 + 3 = 5 rows
            assert len(rows) == 5


class TestClustersJSON:
    def test_export_clusters_json(self, clusters: list[list[str]], tmp_path: Path) -> None:
        path = tmp_path / "clusters.json"
        export_clusters_json(clusters, path)

        with path.open() as f:
            data = json.load(f)
        assert len(data) == 5
        assert data[0]["cluster_id"] == 0
        assert data[0]["record_id"] == "1"


class TestExportFromDataFrame:
    def test_export_matches_from_pandas(self, tmp_path: Path) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            [
                {"left_id": "a", "right_id": "b", "score": 0.9, "scores": [0.85]},
            ]
        )
        path = tmp_path / "matches_df.json"
        export_matches_json(df, path)

        with path.open() as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["left_id"] == "a"
