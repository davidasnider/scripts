"""Command-line entry point for the file catalog application."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def scan(
    root: Path = typer.Argument(..., help="Root directory to catalog."),
    manifest: Path = typer.Option(
        Path("data/manifest.json"),
        "--manifest",
        help="Path to the manifest JSON file (default: data/manifest.json).",
    ),
) -> None:
    """Placeholder implementation for the scan command."""
    logger.info("Scanning %s and writing manifest to %s", root, manifest)
    print(f"Scanning {root} and writing manifest to {manifest}")


@app.command()
def analyze(
    manifest: Path = typer.Option(
        Path("data/manifest.json"),
        "--manifest",
        help="Path to the manifest JSON file (default: data/manifest.json).",
    ),
) -> None:
    """Placeholder implementation for the analyze command."""
    logger.info("Analyzing catalog data from %s", manifest)
    print(f"Analyzing catalog data from {manifest}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
