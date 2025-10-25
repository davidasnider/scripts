#!/usr/bin/env python
# scripts/query_manifest.py
"""A script to query the manifest file."""

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schema import FileRecord

app = typer.Typer()
console = Console()


@app.command()
def query(
    manifest_path: Path = typer.Option(
        "data/manifest.json",
        "--manifest-path",
        "-p",
        help="Path to the manifest.json file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    no_summary: bool = typer.Option(
        False, "--no-summary", help="Find files with no summary."
    ),
    is_nsfw: bool = typer.Option(
        False, "--is-nsfw", help="Find files flagged as NSFW."
    ),
):
    """
    Query the manifest for files based on specific criteria.
    """
    try:
        with manifest_path.open("r") as f:
            manifest_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        console.print(f"[bold red]Error reading manifest file: {e}[/bold red]")
        raise typer.Exit(code=1)

    records = [FileRecord(**record) for record in manifest_data]
    filtered_records: list[FileRecord] = []

    for record in records:
        match = True
        if no_summary and record.summary is not None:
            match = False
        if is_nsfw and not record.is_nsfw:
            match = False

        if match:
            filtered_records.append(record)

    # Convert Pydantic models to a list of dicts for JSON serialization
    output_data: list[dict[str, Any]] = [
        record.model_dump() for record in filtered_records
    ]

    console.print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    app()
