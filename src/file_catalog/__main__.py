"""Command-line entry point for the file catalog application."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="file-catalog",
        description=(
            "Catalog and analyze local files using OCR, embeddings,"
            " and LLM assistance."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan a directory and build or update the file manifest database.",
    )
    scan_parser.add_argument(
        "root",
        type=Path,
        help="Root directory to catalog.",
    )
    scan_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.json"),
        help="Path to the manifest JSON file (default: data/manifest.json).",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run analysis tasks against the catalogued files.",
    )
    analyze_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.json"),
        help="Path to the manifest JSON file (default: data/manifest.json).",
    )

    return parser


def handle_scan(root: Path, manifest: Path) -> None:
    """Placeholder implementation for the scan command."""
    print(f"[TODO] Scanning {root} and writing manifest to {manifest}.")


def handle_analyze(manifest: Path) -> None:
    """Placeholder implementation for the analyze command."""
    print(f"[TODO] Analyzing catalog data from {manifest}.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        handle_scan(args.root, args.manifest)
        return 0

    if args.command == "analyze":
        handle_analyze(args.manifest)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
