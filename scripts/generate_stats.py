#!/usr/bin/env python
# scripts/generate_stats.py
"""A script to generate statistics from the manifest file."""

import json
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Add project root to path to allow importing from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schema import FileRecord  # noqa: E402

app = typer.Typer()
console = Console()
error_console = Console(stderr=True)


class SortOptions(str, Enum):
    mime_type = "mime type"
    total = "total"
    with_text = "with text"
    without_text = "without text"


@app.command()
def generate_stats(
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
    sort_by: Optional[SortOptions] = typer.Option(
        None,
        "--sort-by",
        "-s",
        help="Sort the MIME type table by a specific column.",
        case_sensitive=False,
        show_choices=True,
        rich_help_panel="Sorting Options",
    ),
):
    """
    Generate statistics from the manifest file.
    """
    try:
        with manifest_path.open("r") as f:
            manifest_data = json.load(f)
    except IOError as e:
        console.print(f"[bold red]Failed to read manifest file: {e}[/bold red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Invalid JSON format in manifest file: {e}[/bold red]")
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

    # --- Statistics Calculation ---
    stats = {
        "total_files": len(records),
        "files_by_status": defaultdict(int),
        "completed_files_stats": {
            "total": 0,
            "files_by_mime_type": {},
            "nsfw_files": 0,
            "images_without_text": 0,
            "files_with_summaries": 0,
            "files_without_summaries": 0,
            "files_with_financial_red_flags": 0,
            "files_with_passwords": 0,
            "files_with_estate_relevant_info": 0,
        },
    }

    for record in records:
        stats["files_by_status"][record.status] += 1

        if record.status == "complete":
            completed_stats = stats["completed_files_stats"]
            completed_stats["total"] += 1

            mime_type = record.mime_type
            if mime_type not in completed_stats["files_by_mime_type"]:
                completed_stats["files_by_mime_type"][mime_type] = {
                    "with_text": 0,
                    "without_text": 0,
                    "total": 0,
                }

            mime_stats = completed_stats["files_by_mime_type"][mime_type]
            mime_stats["total"] += 1
            if record.extracted_text:
                mime_stats["with_text"] += 1
            else:
                mime_stats["without_text"] += 1

            if record.is_nsfw:
                completed_stats["nsfw_files"] += 1

            if mime_type.startswith("image/") and not record.extracted_text:
                completed_stats["images_without_text"] += 1

    # --- Statistics Output ---
    console.print("[bold]File Statistics[/bold]")
    console.print(f"Total files: {stats['total_files']}")

    status_table = Table(title="Files by Status")
    status_table.add_column("Status", style="cyan")
    status_table.add_column("Count", style="magenta")
    for status, count in stats["files_by_status"].items():
        status_table.add_row(status, str(count))
    console.print(status_table)

    completed_stats = stats["completed_files_stats"]
    console.print(
        f"\n[bold]Statistics for {completed_stats['total']} completed files:[/bold]"
    )

    # --- Sorting Logic ---
    mime_type_items = completed_stats["files_by_mime_type"].items()
    if sort_by:
        if sort_by == "mime type":
            mime_type_items = sorted(mime_type_items, key=lambda item: item[0])
        elif sort_by == "total":
            mime_type_items = sorted(
                mime_type_items, key=lambda item: item[1]["total"], reverse=True
            )
        elif sort_by == "with text":
            mime_type_items = sorted(
                mime_type_items, key=lambda item: item[1]["with_text"], reverse=True
            )
        elif sort_by == "without text":
            mime_type_items = sorted(
                mime_type_items, key=lambda item: item[1]["without_text"], reverse=True
            )

    mime_table = Table(title="Files by MIME Type")
    mime_table.add_column("MIME Type", style="cyan")
    mime_table.add_column("Total", style="magenta")
    mime_table.add_column("With Text", style="green")
    mime_table.add_column("Without Text", style="red")
    for mime_type, mime_stats in mime_type_items:
        mime_table.add_row(
            mime_type,
            str(mime_stats["total"]),
            str(mime_stats["with_text"]),
            str(mime_stats["without_text"]),
        )
    console.print(mime_table)

    other_stats_table = Table(title="Other Stats")
    other_stats_table.add_column("Statistic", style="cyan")
    other_stats_table.add_column("Count", style="magenta")
    other_stats_table.add_row("NSFW files", str(completed_stats["nsfw_files"]))
    other_stats_table.add_row(
        "Images without text", str(completed_stats["images_without_text"])
    )
    other_stats_table.add_row(
        "Files with summaries", str(completed_stats["files_with_summaries"])
    )
    other_stats_table.add_row(
        "Files without summaries", str(completed_stats["files_without_summaries"])
    )
    other_stats_table.add_row(
        "Files with financial red flags",
        str(completed_stats["files_with_financial_red_flags"]),
    )
    other_stats_table.add_row(
        "Files with passwords", str(completed_stats["files_with_passwords"])
    )
    other_stats_table.add_row(
        "Files with estate relevant info",
        str(completed_stats["files_with_estate_relevant_info"]),
    )
    console.print(other_stats_table)


if __name__ == "__main__":
    app()
