"""Tests for the reclink CLI."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from reclink.cli import main


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    path = tmp_path / "data.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "city"])
        writer.writerow(["1", "John Smith", "NYC"])
        writer.writerow(["2", "Jon Smyth", "New York"])
        writer.writerow(["3", "Jane Doe", "LA"])
        writer.writerow(["4", "Janet Doe", "Los Angeles"])
    return path


@pytest.fixture
def candidates_file(tmp_path: Path) -> Path:
    """Create a candidates text file."""
    path = tmp_path / "candidates.txt"
    path.write_text("John Smith\nJane Doe\nBob Jones\nAlice Brown\n")
    return path


class TestDedupe:
    def test_dedupe_stdout(self, sample_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["dedupe", "--input", str(sample_csv), "--field", "name", "--threshold", "0.7"])
        captured = capsys.readouterr()
        assert "duplicate pair(s) found" in captured.out

    def test_dedupe_csv_output(self, sample_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "results.csv"
        main(
            [
                "dedupe",
                "--input",
                str(sample_csv),
                "--field",
                "name",
                "--threshold",
                "0.7",
                "--output",
                str(output),
            ]
        )
        assert output.exists()
        with output.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 1
        assert "left_id" in rows[0]

    def test_dedupe_json_output(self, sample_csv: Path, tmp_path: Path) -> None:
        output = tmp_path / "results.json"
        main(
            [
                "dedupe",
                "--input",
                str(sample_csv),
                "--field",
                "name",
                "--threshold",
                "0.7",
                "--output",
                str(output),
                "--format",
                "json",
            ]
        )
        assert output.exists()
        with output.open() as f:
            data = json.load(f)
        assert isinstance(data, list)


class TestLink:
    def test_link_stdout(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        with left.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "name"])
            w.writerow(["L1", "John Smith"])
            w.writerow(["L2", "Jane Doe"])
        with right.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "name"])
            w.writerow(["R1", "Jon Smyth"])
            w.writerow(["R2", "Janet Doe"])
        main(
            [
                "link",
                "--left",
                str(left),
                "--right",
                str(right),
                "--field",
                "name",
                "--threshold",
                "0.7",
            ]
        )
        captured = capsys.readouterr()
        assert "link(s) found" in captured.out


class TestMatch:
    def test_match(self, candidates_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(
            [
                "match",
                "--query",
                "Jon Smith",
                "--candidates-file",
                str(candidates_file),
                "--scorer",
                "jaro_winkler",
            ]
        )
        captured = capsys.readouterr()
        assert "match(es) found" in captured.out
        assert "John Smith" in captured.out


class TestExplain:
    def test_explain(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["explain", "John Smith", "Jon Smyth"])
        captured = capsys.readouterr()
        assert "Comparing" in captured.out
        assert "jaro_winkler" in captured.out


class TestHelp:
    def test_no_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "reclink" in captured.out
