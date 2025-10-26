#!/usr/bin/env python
# scripts/query_manifest.py
"""A script to query the manifest file."""

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import track

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schema import FileRecord  # noqa: E402

app = typer.Typer()
console = Console()
error_console = Console(stderr=True)


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
    no_text: bool = typer.Option(
        False, "--no-text", help="Find files with no extracted text."
    ),
    completed_analysis: bool = typer.Option(
        False, "--completed-analysis", help="Find files with completed analysis."
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

    records = [
        FileRecord(**record)
        for record in track(
            manifest_data,
            description="Loading records...",
            total=len(manifest_data),
            console=error_console,
        )
    ]

    # Early return if no filters are active
    if not (no_summary or is_nsfw or no_text or completed_analysis):
        filtered_records = records
    else:
        # Single-pass filtering using list comprehension
        def matches_criteria(record: FileRecord) -> bool:
            # Check each filter condition
            passes_summary_filter = not no_summary or record.summary is None
            passes_nsfw_filter = not is_nsfw or (record.is_nsfw is True)
            passes_text_filter = not no_text or not record.extracted_text
            passes_completed_analysis_filter = (
                not completed_analysis or record.status == "complete"
            )

            # All active filters must match (AND logic)
            return (
                passes_summary_filter
                and passes_nsfw_filter
                and passes_text_filter
                and passes_completed_analysis_filter
            )

        with error_console.status("[bold green]Filtering records..."):
            filtered_records = [
                record for record in records if matches_criteria(record)
            ]

    # Print summary of filtering results
    error_console.print(
        f"[green]Filtered to {len(filtered_records)} of {len(records)} records.[/green]"
    )

    # Convert Pydantic models to a list of dicts for JSON serialization
    output_data: list[dict[str, Any]] = [
        record.model_dump()
        for record in track(
            filtered_records, description="Formatting output...", console=error_console
        )
    ]

    with error_console.status("[bold green]Serializing to JSON..."):
        json_output = json.dumps(output_data, indent=2)

    console.print(json_output)


if __name__ == "__main__":
    app()
