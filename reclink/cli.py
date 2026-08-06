"""Command-line interface for reclink.

Usage::

    reclink dedupe --input data.csv --field name --threshold 0.85
    reclink match --query "John Smith" --candidates names.txt
    reclink explain "John Smith" "Jon Smyth"
    reclink link --left a.csv --right b.csv --field name
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import reclink


def _read_csv(path: str, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts."""
    with Path(path).open(encoding=encoding) as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(rows: list[dict[str, Any]], path: str, encoding: str = "utf-8") -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cmd_dedupe(args: argparse.Namespace) -> None:
    """Deduplicate records in a CSV file."""
    data = _read_csv(args.input)
    if not data:
        print("No records found in input file.", file=sys.stderr)
        sys.exit(1)

    if args.field not in data[0]:
        print(f"Field '{args.field}' not found. Available: {list(data[0].keys())}", file=sys.stderr)
        sys.exit(1)

    # Build pipeline
    builder = (
        reclink.pipeline.ReclinkPipeline.builder()
        .preprocess(args.field, ["fold_case", "normalize_whitespace"])
        .block_sorted_neighborhood(args.field, window=5)
        .compare_string(args.field, metric=args.scorer)
        .classify_threshold(args.threshold)
    )
    pipeline = builder.build()

    id_column = args.id_column
    if id_column not in data[0]:
        # Auto-assign IDs
        for i, row in enumerate(data):
            row[id_column] = str(i)

    results = pipeline.dedup(data, id_column=id_column)

    output_rows = results if isinstance(results, list) else results.to_dict("records")

    if args.output:
        if args.format == "json":
            with Path(args.output).open("w") as f:
                json.dump(output_rows, f, indent=2, default=str)
        else:
            _write_csv(
                [
                    {
                        "left_id": r["left_id"],
                        "right_id": r["right_id"],
                        "score": r["score"],
                        "match_class": r.get("match_class", ""),
                    }
                    for r in output_rows
                ],
                args.output,
            )
        print(f"Found {len(output_rows)} duplicate pairs. Results written to {args.output}")
    else:
        for r in output_rows:
            print(f"  {r['left_id']} <-> {r['right_id']}  score={r['score']:.4f}")
        print(f"\n{len(output_rows)} duplicate pair(s) found.")


def cmd_link(args: argparse.Namespace) -> None:
    """Link records between two CSV files."""
    left_data = _read_csv(args.left)
    right_data = _read_csv(args.right)

    if not left_data or not right_data:
        print("One or both input files are empty.", file=sys.stderr)
        sys.exit(1)

    if args.field not in left_data[0] or args.field not in right_data[0]:
        print(f"Field '{args.field}' not found in one or both files.", file=sys.stderr)
        sys.exit(1)

    builder = (
        reclink.pipeline.ReclinkPipeline.builder()
        .preprocess(args.field, ["fold_case", "normalize_whitespace"])
        .block_sorted_neighborhood(args.field, window=5)
        .compare_string(args.field, metric=args.scorer)
        .classify_threshold(args.threshold)
    )
    pipeline = builder.build()

    id_column = args.id_column
    for dataset, label in [(left_data, "left"), (right_data, "right")]:
        if id_column not in dataset[0]:
            for i, row in enumerate(dataset):
                row[id_column] = f"{label}_{i}"

    results = pipeline.link(left_data, right_data, id_column=id_column)

    output_rows = results if isinstance(results, list) else results.to_dict("records")

    if args.output:
        if args.format == "json":
            with Path(args.output).open("w") as f:
                json.dump(output_rows, f, indent=2, default=str)
        else:
            _write_csv(
                [
                    {
                        "left_id": r["left_id"],
                        "right_id": r["right_id"],
                        "score": r["score"],
                        "match_class": r.get("match_class", ""),
                    }
                    for r in output_rows
                ],
                args.output,
            )
        print(f"Found {len(output_rows)} links. Results written to {args.output}")
    else:
        for r in output_rows:
            print(f"  {r['left_id']} <-> {r['right_id']}  score={r['score']:.4f}")
        print(f"\n{len(output_rows)} link(s) found.")


def cmd_match(args: argparse.Namespace) -> None:
    """Find best matches for a query string."""
    if args.candidates_file:
        with Path(args.candidates_file).open() as f:
            candidates = [line.strip() for line in f if line.strip()]
    else:
        print("--candidates-file is required.", file=sys.stderr)
        sys.exit(1)

    results = reclink.match_batch(
        args.query,
        candidates,
        scorer=args.scorer,
        threshold=args.threshold,
        limit=args.limit,
    )

    if not results:
        print("No matches found.")
        return

    for matched, score, idx in results:
        print(f"  {score:.4f}  {matched}  (index={idx})")
    print(f"\n{len(results)} match(es) found.")


def cmd_explain(args: argparse.Namespace) -> None:
    """Show per-algorithm score breakdown for a pair of strings."""
    scores = reclink.explain(args.string_a, args.string_b)
    max_name_len = max(len(name) for name in scores) if scores else 0
    print(f"\n  Comparing: {args.string_a!r}  vs  {args.string_b!r}\n")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(score * 30)
        print(f"  {name:<{max_name_len}}  {score:.4f}  {bar}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="reclink",
        description="Blazing-fast fuzzy matching and record linkage CLI.",
    )
    parser.add_argument("--version", action="version", version=f"reclink {reclink.__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # dedupe
    p_dedupe = subparsers.add_parser("dedupe", help="Deduplicate records in a CSV file")
    p_dedupe.add_argument("--input", "-i", required=True, help="Input CSV file path")
    p_dedupe.add_argument("--field", "-f", required=True, help="Field to match on")
    p_dedupe.add_argument("--threshold", "-t", type=float, default=0.85, help="Match threshold")
    p_dedupe.add_argument("--scorer", "-s", default="jaro_winkler", help="Similarity metric")
    p_dedupe.add_argument("--id-column", default="id", help="ID column name")
    p_dedupe.add_argument("--output", "-o", help="Output file path")
    p_dedupe.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")

    # link
    p_link = subparsers.add_parser("link", help="Link records between two CSV files")
    p_link.add_argument("--left", "-l", required=True, help="Left CSV file path")
    p_link.add_argument("--right", "-r", required=True, help="Right CSV file path")
    p_link.add_argument("--field", "-f", required=True, help="Field to match on")
    p_link.add_argument("--threshold", "-t", type=float, default=0.85, help="Match threshold")
    p_link.add_argument("--scorer", "-s", default="jaro_winkler", help="Similarity metric")
    p_link.add_argument("--id-column", default="id", help="ID column name")
    p_link.add_argument("--output", "-o", help="Output file path")
    p_link.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")

    # match
    p_match = subparsers.add_parser("match", help="Find best matches for a query string")
    p_match.add_argument("--query", "-q", required=True, help="Query string")
    p_match.add_argument(
        "--candidates-file",
        "-c",
        required=True,
        help="Candidate strings file (one per line)",
    )
    p_match.add_argument("--threshold", "-t", type=float, default=None, help="Minimum score")
    p_match.add_argument("--scorer", "-s", default="jaro_winkler", help="Similarity metric")
    p_match.add_argument("--limit", "-n", type=int, default=10, help="Max results")

    # explain
    p_explain = subparsers.add_parser("explain", help="Show score breakdown for two strings")
    p_explain.add_argument("string_a", help="First string")
    p_explain.add_argument("string_b", help="Second string")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "dedupe": cmd_dedupe,
        "link": cmd_link,
        "match": cmd_match,
        "explain": cmd_explain,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
