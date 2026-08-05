"""Export utilities for record linkage results.

Write match results and clusters to CSV or JSON using stdlib only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _results_to_rows(results: Any) -> list[dict[str, Any]]:
    """Convert results (list of dicts or DataFrame) to list of dicts."""
    if hasattr(results, "to_dicts"):
        return list(results.to_dicts())
    if hasattr(results, "to_dict"):
        return list(results.to_dict("records"))
    if isinstance(results, list):
        return results
    raise TypeError(f"Unsupported results type: {type(results)}")


def export_matches_csv(
    results: Any,
    path: str | Path,
) -> None:
    """Write match results to CSV.

    Parameters
    ----------
    results : list of dict or DataFrame
        Match results with 'left_id', 'right_id', 'score', and 'scores' keys.
    path : str or Path
        Output file path.
    """
    rows = _results_to_rows(results)
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_id", "right_id", "score", "scores"])
        for row in rows:
            scores_str = ";".join(str(s) for s in row.get("scores", []))
            writer.writerow([row["left_id"], row["right_id"], row["score"], scores_str])


def export_matches_json(
    results: Any,
    path: str | Path,
) -> None:
    """Write match results to JSON.

    Parameters
    ----------
    results : list of dict or DataFrame
        Match results with 'left_id', 'right_id', 'score', and 'scores' keys.
    path : str or Path
        Output file path.
    """
    rows = _results_to_rows(results)
    path = Path(path)
    with path.open("w") as f:
        json.dump(rows, f, indent=2, default=str)


def export_clusters_csv(
    clusters: Any,
    path: str | Path,
) -> None:
    """Write clusters to CSV with cluster_id and record_id columns.

    Parameters
    ----------
    clusters : list of list of str, or DataFrame
        Cluster groups. If DataFrame, expects 'cluster_id' and 'record_id' columns.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "record_id"])
        if hasattr(clusters, "to_dicts"):
            for row in clusters.to_dicts():
                writer.writerow([row["cluster_id"], row["record_id"]])
        elif hasattr(clusters, "to_dict"):
            for row in clusters.to_dict("records"):
                writer.writerow([row["cluster_id"], row["record_id"]])
        elif isinstance(clusters, list):
            for cluster_id, group in enumerate(clusters):
                for record_id in group:
                    writer.writerow([cluster_id, record_id])
        else:
            raise TypeError(f"Unsupported clusters type: {type(clusters)}")


def export_clusters_json(
    clusters: Any,
    path: str | Path,
) -> None:
    """Write clusters to JSON.

    Parameters
    ----------
    clusters : list of list of str, or DataFrame
        Cluster groups. If DataFrame, expects 'cluster_id' and 'record_id' columns.
    path : str or Path
        Output file path.
    """
    path = Path(path)

    if hasattr(clusters, "to_dicts"):
        data = clusters.to_dicts()
    elif hasattr(clusters, "to_dict"):
        data = clusters.to_dict("records")
    elif isinstance(clusters, list):
        data = [
            {"cluster_id": i, "record_id": rid} for i, group in enumerate(clusters) for rid in group
        ]
    else:
        raise TypeError(f"Unsupported clusters type: {type(clusters)}")

    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
