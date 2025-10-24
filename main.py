#!/usr/bin/env python3
"""Main orchestrator for the file catalog pipeline."""

from __future__ import annotations

import math
import os
import tempfile

# Disable tokenizers parallelism to avoid warnings in threaded environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import logging
import queue
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from shutil import copy2
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import pandas as pd
import typer
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

from src.access_analysis import analyze_access_database
from src.ai_analyzer import (
    TEXT_ANALYZER_CHUNK_TOKENS,
    analyze_estate_relevant_information,
    analyze_financial_document,
    analyze_text_content,
    describe_image,
    detect_passwords,
    summarize_video_frames,
)
from src.content_extractor import (
    extract_content_from_docx,
    extract_content_from_image,
    extract_content_from_pdf,
    extract_content_from_xlsx,
    extract_frames_from_video,
)
from src.database_manager import add_file_to_db, initialize_db
from src.logging_utils import configure_logging
from src.manifest_utils import (
    reset_file_record_for_rescan,
    reset_outdated_analysis_tasks,
)
from src.nsfw_classifier import NSFWClassifier
from src.schema import (
    COMPLETE,
    FAILED,
    PENDING_ANALYSIS,
    PENDING_EXTRACTION,
    AnalysisName,
    AnalysisStatus,
    FileRecord,
)
from src.task_utils import ensure_required_tasks
from src.text_utils import count_tokens


class AnalysisModel(str, Enum):
    """Enum for the analysis models."""

    TEXT_ANALYZER = "text_analysis"
    PEOPLE_ANALYZER = "people_analysis"
    CODE_ANALYZER = "code_analysis"
    IMAGE_DESCRIBER = "image_description"
    VIDEO_ANALYZER = "video_summary"
    ALL = "all"


@dataclass
class WorkItem:
    """Work item for queue-based processing."""

    file_record: FileRecord
    correlation_id: str


@dataclass
class ActiveFileStatus:
    """Track live progress information for an in-flight file."""

    file_name: str
    stage: str
    current_task: str | None = None
    chunks_processed: int = 0
    chunks_total: int | None = None
    tasks_completed: int = 0
    tasks_total: int = 0


# Constants
MANIFEST_PATH = Path("data/manifest.json")
DB_PATH = Path("data/chromadb")
MANIFEST_BACKUP_DIR = MANIFEST_PATH.parent / "manifest_backups"
MANIFEST_BACKUP_LIMIT = 20

TEXT_BASED_ANALYSES = {
    AnalysisName.TEXT_ANALYSIS,
    AnalysisName.PEOPLE_ANALYSIS,
    AnalysisName.ESTATE_ANALYSIS,
    AnalysisName.PASSWORD_DETECTION,
}
TEXT_BASED_ANALYSIS_MODEL_VALUES = {analysis.value for analysis in TEXT_BASED_ANALYSES}
# avoids treating random bytes as content during strings-style fallback
MIN_ASCII_CHUNK_LENGTH = 4


# Threading configuration
NUM_EXTRACTION_WORKERS = 5
NUM_ANALYSIS_WORKERS = (
    2  # AI analysis remains a bottleneck; keep a few dedicated threads
)
NUM_DATABASE_WORKERS = 1
MINIMUM_WORKER_TOTAL = 3

# Shared state
extraction_queue = queue.Queue()
analysis_queue = queue.Queue()
database_queue = queue.Queue()
completed_files = set()
failed_files = set()
in_progress_files: dict[str, ActiveFileStatus] = {}
lock = threading.Lock()
shutdown_event = threading.Event()
_shutdown_signals_sent = threading.Event()
manifest_write_lock = threading.Lock()

CHUNK_PROGRESS_LOG_PATTERN = re.compile(
    (
        r"^(?P<task>.+?) chunk (?P<current>\d+)/(?P<total>\d+) for "
        r"(?P<file>.+?) \((?P<remaining>\d+) remaining\)$"
    ),
    re.IGNORECASE,
)

# Active worker counts (updated when pipeline runs)
_active_worker_counts = {
    "extraction": NUM_EXTRACTION_WORKERS,
    "analysis": NUM_ANALYSIS_WORKERS,
    "database": NUM_DATABASE_WORKERS,
}

# Global manifest for signal handlers
current_manifest = None

LOGGER_NAME = "file_catalog.pipeline"
pipeline_logger = logging.getLogger(LOGGER_NAME)
extraction_logger = logging.getLogger(f"{LOGGER_NAME}.extraction")
analysis_logger = logging.getLogger(f"{LOGGER_NAME}.analysis")
database_logger = logging.getLogger(f"{LOGGER_NAME}.database")

LIVE_CONSOLE = Console()
LOG_PANEL_RATIO = 0.5
TOP_PANEL_ROWS = 31
CHUNK_TOKEN_LIMIT = TEXT_ANALYZER_CHUNK_TOKENS
SMALL_TEXT_BYTE_THRESHOLD = 3000
MAX_BACKUP_SUFFIX_ATTEMPTS = 100
ACTIVE_STATUS_FIELDS = {
    "file_name",
    "stage",
    "current_task",
    "chunks_processed",
    "chunks_total",
    "tasks_completed",
    "tasks_total",
}
SHUTDOWN_JOIN_TIMEOUT = 2.0
NORMAL_JOIN_TIMEOUT = 5.0


def _calculate_log_panel_display_limit(
    console: Console,
    *,
    ratio: float = LOG_PANEL_RATIO,
) -> int:
    """
    Return log panel line budget based on the current terminal height.

    Args:
        console (Console): The Rich Console instance to query for terminal size.
        ratio (float, optional): Fraction of terminal height for log panel.
            Should be between 0.0 and 1.0. Defaults to LOG_PANEL_RATIO.

    Returns:
        int: Number of lines to display in the log panel.
    """

    terminal_height = max(console.size.height, 1)
    return max(int(terminal_height * ratio) - 1, 1)


def _join_threads_with_timeout(
    threads: Iterable[threading.Thread], timeout: float
) -> list[str]:
    """Join threads for up to timeout seconds, returning names still alive."""

    lingering: list[str] = []
    deadline = time.monotonic() + max(timeout, 0)
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            lingering.append(thread.name or repr(thread))
            continue
        thread.join(remaining)
        if thread.is_alive():
            lingering.append(thread.name or repr(thread))
    return lingering


@dataclass
class ChunkMetrics:
    """Aggregated timing information for LLM chunk processing."""

    total_chunks: int = 0
    total_duration: float = 0.0
    file_samples: int = 0
    recent_file_chunk_averages: deque[float] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    recent_file_chunk_counts: deque[int] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    last_file_average: float | None = None
    last_file_chunks: int = 0
    last_updated: float | None = None


chunk_metrics = ChunkMetrics()
# All reads/writes of chunk_metrics must hold `lock` to keep metrics atomic.


def _update_active_file_status(
    correlation_id: str, logger: logging.Logger, **kwargs: Any
) -> None:
    """Safely merge ActiveFileStatus updates for a given correlation ID."""

    invalid = [key for key in kwargs if key not in ACTIVE_STATUS_FIELDS]
    if invalid:
        logger.debug(
            "Ignoring invalid active status fields for %s: %s",
            correlation_id,
            ", ".join(sorted(invalid)),
        )
    with lock:
        status = in_progress_files.get(correlation_id)
        if not status:
            return
        for key, value in kwargs.items():
            if key in ACTIVE_STATUS_FIELDS:
                setattr(status, key, value)


def _increment_active_chunks_total(
    correlation_id: str | None, *, increment: int
) -> None:
    """Increase the planned chunk total for an active file, if present."""

    if increment <= 0 or not correlation_id:
        return
    with lock:
        status = in_progress_files.get(correlation_id)
        if not status:
            return
        current_total = status.chunks_total or 0
        status.chunks_total = current_total + increment


def _format_task_label(label: str) -> str:
    """Normalize task labels emitted in log lines."""

    cleaned = " ".join(label.strip().split())
    return cleaned.title() if cleaned else ""


def _apply_chunk_progress_from_log(message: str) -> bool:
    """Update active file chunk progress based on a chunk progress log line."""

    text = message.strip()
    if not text:
        return False

    match = CHUNK_PROGRESS_LOG_PATTERN.match(text)
    if not match:
        return False

    try:
        current = int(match.group("current"))
        total = int(match.group("total"))
    except (TypeError, ValueError):
        return False

    if total <= 0:
        return False

    file_name = (match.group("file") or "").strip()
    task_label = (match.group("task") or "").strip()
    if not file_name or not task_label:
        return False

    normalized_task = _format_task_label(task_label)
    clamped_current = max(0, min(current, total))
    updated = False

    with lock:
        for status in in_progress_files.values():
            if status.file_name.lower() != file_name.lower():
                continue
            status.stage = "Analyzing"
            if normalized_task:
                status.current_task = normalized_task
            status.chunks_total = total
            status.chunks_processed = clamped_current
            updated = True

    return updated


@dataclass
class FileChunkProgress:
    """Mutable accumulator tracking per-file chunk usage."""

    count: int = 0
    duration: float = 0.0

    def add(self, *, chunk_count: int, duration: float) -> bool:
        if chunk_count <= 0 or duration < 0:
            return False
        self.count += chunk_count
        self.duration += duration
        return True


def _format_active_files(active: list[ActiveFileStatus], limit: int = 3) -> str:
    """Build a short preview of active files for logging."""

    if not active:
        return ""
    preview_parts: list[str] = []
    for status in active[:limit]:
        details: list[str] = [status.stage]
        if status.current_task:
            details.append(status.current_task)
        chunk_status = _format_chunk_progress(status)
        if chunk_status != "—":
            details.append(chunk_status)
        task_status = _format_task_progress(status)
        if task_status != "—":
            details.append(f"tasks {task_status}")
        detail_text = "; ".join(details)
        preview_parts.append(f"{status.file_name} ({detail_text})")
    if len(active) > limit:
        preview_parts.append("...")
    return ", ".join(preview_parts)


def _resolve_worker_counts(
    max_threads: int | None,
) -> tuple[int, int, int, int | None, bool]:
    """
    Determine worker thread counts after applying an optional max_threads limit.

    Returns:
        tuple: extraction_workers, analysis_workers, database_workers,
        applied_limit (or None if unused), limit_was_increased (bool).
    """

    extraction_workers = NUM_EXTRACTION_WORKERS
    analysis_workers = NUM_ANALYSIS_WORKERS
    database_workers = NUM_DATABASE_WORKERS

    if not max_threads or max_threads <= 0:
        return extraction_workers, analysis_workers, database_workers, None, False

    applied_limit = max(max_threads, MINIMUM_WORKER_TOTAL)
    limit_was_increased = applied_limit != max_threads

    worker_counts = {
        "analysis": analysis_workers,
        "extraction": extraction_workers,
        "database": database_workers,
    }
    total_workers = sum(worker_counts.values())

    while total_workers > applied_limit:
        for key in ("analysis", "extraction", "database"):
            min_allowed = 1
            if worker_counts[key] > min_allowed:
                worker_counts[key] -= 1
                total_workers -= 1
                break
        else:
            break

    return (
        worker_counts["extraction"],
        worker_counts["analysis"],
        worker_counts["database"],
        applied_limit,
        limit_was_increased,
    )


class ShutdownRequested(Exception):
    """Raised to abort processing when a shutdown has been requested."""


def _check_for_shutdown() -> None:
    if shutdown_event.is_set():
        raise ShutdownRequested


def dispatch_shutdown_to_workers() -> None:
    """Send sentinel values to worker queues exactly once."""

    if _shutdown_signals_sent.is_set():
        return

    _shutdown_signals_sent.set()
    for worker_count, queue_ref in (
        (_active_worker_counts["extraction"], extraction_queue),
        (_active_worker_counts["analysis"], analysis_queue),
        (_active_worker_counts["database"], database_queue),
    ):
        for _ in range(worker_count):
            queue_ref.put(None)


def request_shutdown(reason: str, *, raise_interrupt: bool = False) -> None:
    """Trigger a coordinated shutdown of the pipeline."""

    if shutdown_event.is_set():
        pipeline_logger.warning(
            "Additional shutdown request received (%s); forcing termination.",
            reason,
        )
        if raise_interrupt:
            os._exit(1)
        return

    pipeline_logger.info("Shutdown requested: %s", reason)
    shutdown_event.set()
    dispatch_shutdown_to_workers()

    if current_manifest is not None:
        try:
            save_manifest(current_manifest)
            pipeline_logger.info("Manifest saved during shutdown request")
        except Exception as exc:
            pipeline_logger.error(
                "Failed to save manifest during shutdown request: %s", exc
            )

    if raise_interrupt:
        threading.interrupt_main()


@dataclass(frozen=True)
class ChunkMetricsSnapshot:
    """Immutable snapshot of aggregated LLM chunk timings."""

    total_chunks: int
    avg_chunk_seconds: float | None
    recent_avg_seconds: float | None
    avg_chunks_per_file: float | None
    file_samples: int
    estimated_remaining_chunks: float | None
    estimated_remaining_seconds: float | None
    last_file_chunks: int | None
    last_file_avg_seconds: float | None
    recent_chunk_counts: list[int]
    last_updated: float | None
    timeout_p75: float | None


@dataclass(frozen=True)
class ProgressSnapshot:
    """Point-in-time summary of pipeline activity for the live dashboard."""

    total: int
    completed: int
    failed: int
    remaining: int
    active_files: list[ActiveFileStatus]
    elapsed_seconds: float
    start_timestamp: float
    model_label: str
    chunk_limit_label: str
    chunk_metrics: ChunkMetricsSnapshot
    queue_sizes: dict[str, int | None]
    worker_counts: dict[str, int]


def _estimate_chunk_count(text: str | None, *, max_chunks: int | None) -> int:
    """Estimate how many LLM chunks will be processed for the given text."""

    if not text:
        return 0

    text_bytes = len(text.encode("utf-8"))
    if text_bytes <= SMALL_TEXT_BYTE_THRESHOLD:
        return 1

    token_count = count_tokens(text)
    if token_count <= 0:
        return 1

    chunks_estimate = max(
        math.ceil(token_count / CHUNK_TOKEN_LIMIT),
        1,
    )
    if max_chunks and max_chunks > 0:
        chunks_estimate = min(chunks_estimate, max_chunks)
    return chunks_estimate


def _resolve_chunk_metric(actual_chunks: Any, chunk_estimate: int) -> int:
    """Prefer a reported chunk count when available, otherwise fall back to estimate."""

    if isinstance(actual_chunks, int) and actual_chunks >= 0:
        return actual_chunks
    return chunk_estimate


def _record_file_chunk_metrics(duration: float, chunk_count: int) -> None:
    """Persist aggregated chunk timing information for a processed file."""

    if chunk_count <= 0 or duration < 0:
        return

    per_chunk = duration / chunk_count if duration else 0.0
    timestamp = time.time()

    with lock:
        chunk_metrics.total_chunks += chunk_count
        chunk_metrics.total_duration += duration
        chunk_metrics.file_samples += 1
        chunk_metrics.recent_file_chunk_averages.append(per_chunk)
        chunk_metrics.recent_file_chunk_counts.append(chunk_count)
        chunk_metrics.last_file_average = per_chunk
        chunk_metrics.last_file_chunks = chunk_count
        chunk_metrics.last_updated = timestamp


def _get_chunk_metrics_snapshot(remaining_files: int) -> ChunkMetricsSnapshot:
    """Return a thread-safe snapshot of current chunk timing statistics."""

    with lock:
        total_chunks = chunk_metrics.total_chunks
        total_duration = chunk_metrics.total_duration
        file_samples = chunk_metrics.file_samples
        recent_avgs = list(chunk_metrics.recent_file_chunk_averages)
        recent_counts = list(chunk_metrics.recent_file_chunk_counts)
        last_avg = chunk_metrics.last_file_average
        last_chunks = chunk_metrics.last_file_chunks or None
        last_updated = chunk_metrics.last_updated

    avg_chunk_seconds = total_duration / total_chunks if total_chunks > 0 else None
    avg_chunks_per_file = total_chunks / file_samples if file_samples > 0 else None
    recent_avg_seconds = sum(recent_avgs) / len(recent_avgs) if recent_avgs else None

    estimated_remaining_chunks: float | None = None
    estimated_remaining_seconds: float | None = None

    if avg_chunks_per_file is not None:
        estimated_remaining_chunks = avg_chunks_per_file * max(remaining_files, 0)
    if estimated_remaining_chunks is not None and avg_chunk_seconds is not None:
        estimated_remaining_seconds = estimated_remaining_chunks * avg_chunk_seconds

    timeout_p75: float | None = None
    positive_avgs = [value for value in recent_avgs if value > 0]
    if positive_avgs:
        positive_avgs.sort()
        percentile_index = math.ceil(0.75 * (len(positive_avgs) + 1)) - 1
        percentile_index = min(
            max(percentile_index, 0),
            len(positive_avgs) - 1,
        )
        timeout_p75 = positive_avgs[percentile_index]

    return ChunkMetricsSnapshot(
        total_chunks=total_chunks,
        avg_chunk_seconds=avg_chunk_seconds,
        recent_avg_seconds=recent_avg_seconds,
        avg_chunks_per_file=avg_chunks_per_file,
        file_samples=file_samples,
        estimated_remaining_chunks=estimated_remaining_chunks,
        estimated_remaining_seconds=estimated_remaining_seconds,
        last_file_chunks=last_chunks,
        last_file_avg_seconds=last_avg,
        recent_chunk_counts=recent_counts,
        last_updated=last_updated,
        timeout_p75=timeout_p75,
    )


def _safe_queue_size(queue_ref: queue.Queue) -> int | None:
    """Best-effort retrieval of a queue's size (may not be supported)."""

    try:
        return queue_ref.qsize()
    except NotImplementedError:
        return None


def _format_duration(seconds: float | None) -> str:
    """Return a human-friendly duration string."""

    if seconds is None:
        return "—"
    remaining = int(max(seconds, 0))
    hours, remainder = divmod(remaining, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts or secs or remaining == 0:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _format_seconds(seconds: float | None) -> str:
    """Format a floating-point duration in seconds."""

    if seconds is None:
        return "—"
    return f"{seconds:.1f}s"


def _format_rate(value: float | None, unit: str) -> str:
    """Format a rate measurement with one decimal place."""

    if value is None or value <= 0:
        return "—"
    if unit:
        return f"{value:.1f} {unit}"
    return f"{value:.1f}"


def _format_eta(seconds: float | None) -> str:
    """Format an ETA string that includes the absolute completion time."""

    if seconds is None:
        return "Collecting"
    if seconds <= 0:
        return "Any moment now"

    return _format_duration(seconds)


def _format_time_since(timestamp: float | None) -> str:
    """Format how long ago a timestamp occurred."""

    if timestamp is None:
        return "—"
    delta = max(time.time() - timestamp, 0.0)
    return _format_duration(delta)


def _collect_progress_snapshot(
    *,
    total: int,
    completed: int,
    failed: int,
    remaining: int,
    active: list[ActiveFileStatus],
    start_time: float,
    model: AnalysisModel,
    chunk_limit: int | None,
) -> ProgressSnapshot:
    """Gather the renderable progress snapshot for the dashboard."""

    elapsed = max(time.time() - start_time, 0.0)
    chunk_snapshot = _get_chunk_metrics_snapshot(remaining)
    queue_sizes = {
        "extraction": _safe_queue_size(extraction_queue),
        "analysis": _safe_queue_size(analysis_queue),
        "database": _safe_queue_size(database_queue),
    }
    worker_counts = dict(_active_worker_counts)

    chunk_limit_label = str(chunk_limit) if chunk_limit is not None else "Unlimited"
    active_snapshot = [replace(item) for item in active]

    return ProgressSnapshot(
        total=total,
        completed=completed,
        failed=failed,
        remaining=remaining,
        active_files=active_snapshot,
        elapsed_seconds=elapsed,
        start_timestamp=start_time,
        model_label=model.value,
        chunk_limit_label=chunk_limit_label,
        chunk_metrics=chunk_snapshot,
        queue_sizes=queue_sizes,
        worker_counts=worker_counts,
    )


def _format_chunk_progress(status: ActiveFileStatus) -> str:
    """Format chunk progress for display."""

    processed = status.chunks_processed
    total = status.chunks_total
    if total and total > 0:
        return f"{processed}/{total}"
    if processed > 0:
        return str(processed)
    return "—"


def _format_task_progress(status: ActiveFileStatus) -> str:
    """Format analysis task progress for display."""

    total = status.tasks_total
    completed = status.tasks_completed
    if total > 0:
        return f"{completed}/{total}"
    return "—"


def _render_active_files_panel(active_files: list[ActiveFileStatus]) -> Panel:
    """Render the active files section."""

    table = Table(
        expand=True,
        show_header=True,
        header_style="bold cyan",
        pad_edge=False,
    )
    table.add_column("File", justify="left", overflow="fold")
    table.add_column("Stage", justify="left", no_wrap=True)
    table.add_column("Tasks", justify="left", no_wrap=True)
    table.add_column("Task", justify="left", overflow="fold")
    table.add_column("Chunks", justify="left")

    if active_files:
        max_display = 5
        for status in active_files[:max_display]:
            table.add_row(
                status.file_name,
                status.stage,
                _format_task_progress(status),
                status.current_task or "—",
                _format_chunk_progress(status),
            )
        if len(active_files) > max_display:
            table.add_row(
                f"…and {len(active_files) - max_display} more",
                "",
                "",
                "",
                "",
            )
    else:
        table.add_row("Pipeline idle", "", "", "", "")

    return Panel(table, title="Active Files", border_style="yellow")


def _render_progress_panel(snapshot: ProgressSnapshot) -> Panel:
    """Render the rich status panel for the top section of the layout."""

    processed = snapshot.completed + snapshot.failed
    total = max(snapshot.total, processed)
    percent_complete = (processed / total * 100) if total else 0.0

    bar_total = total or 1
    bar_end = min(processed, bar_total)
    bar_width = 40
    fill_ratio = bar_end / bar_total if bar_total else 0.0
    filled_chars = max(0, min(bar_width, int(round(fill_ratio * bar_width))))
    empty_chars = bar_width - filled_chars

    outline_top = Text("╭" + "─" * bar_width + "╮", style="magenta")
    outline_bottom = Text("╰" + "─" * bar_width + "╯", style="magenta")

    def _build_fill_line() -> Text:
        line = Text("│", style="magenta")
        if filled_chars:
            line.append("█" * filled_chars, style="magenta")
        if empty_chars:
            line.append(" " * empty_chars, style="grey30")
        line.append("│", style="magenta")
        return line

    fill_line_top = _build_fill_line()
    fill_line_bottom = _build_fill_line()
    progress_details = Text(
        f"{percent_complete:.1f}% complete • {processed}/{total} files",
        style="bold magenta",
    )
    progress_column = Group(
        outline_top,
        fill_line_top,
        fill_line_bottom,
        outline_bottom,
        progress_details,
    )

    files_per_minute = None
    if snapshot.elapsed_seconds > 0 and processed > 0:
        files_per_minute = processed / (snapshot.elapsed_seconds / 60)
    summary_metrics: list[tuple[str, str]] = [
        ("Completed", str(snapshot.completed)),
        ("Failed", str(snapshot.failed)),
        ("Remaining", str(max(snapshot.remaining, 0))),
        ("Elapsed", _format_duration(snapshot.elapsed_seconds)),
        ("Files/min", _format_rate(files_per_minute, "")),
        ("LLM ETA", _format_eta(snapshot.chunk_metrics.estimated_remaining_seconds)),
    ]
    if len(summary_metrics) % 2:
        summary_metrics.append(("", ""))

    summary_table = Table.grid(expand=True, padding=(0, 3))
    summary_table.add_column(justify="left", ratio=1)
    summary_table.add_column(justify="left", ratio=1)
    for index in range(0, len(summary_metrics), 2):
        left_label, left_value = summary_metrics[index]
        right_label, right_value = summary_metrics[index + 1]
        left_cell = f"[bold]{left_label}:[/bold] {left_value}" if left_label else ""
        right_cell = f"[bold]{right_label}:[/bold] {right_value}" if right_label else ""
        summary_table.add_row(left_cell, right_cell)

    progress_grid = Table.grid(expand=True, padding=(0, 2))
    progress_grid.add_column(ratio=2)
    progress_grid.add_column(ratio=1)
    progress_grid.add_row(progress_column, summary_table)

    llm_metrics: list[tuple[str, str]] = []
    llm_metrics.append(("Chunks processed", str(snapshot.chunk_metrics.total_chunks)))
    llm_metrics.append(
        ("Avg chunk", _format_seconds(snapshot.chunk_metrics.avg_chunk_seconds))
    )
    llm_metrics.append(
        ("Recent avg", _format_seconds(snapshot.chunk_metrics.recent_avg_seconds))
    )
    llm_metrics.append(
        (
            "Chunks/file",
            _format_rate(snapshot.chunk_metrics.avg_chunks_per_file, "per file"),
        )
    )
    chunk_rate_per_minute = None
    if snapshot.chunk_metrics.avg_chunk_seconds:
        chunk_rate_per_minute = 60 / snapshot.chunk_metrics.avg_chunk_seconds
    llm_metrics.append(("Chunks/min", _format_rate(chunk_rate_per_minute, "per min")))
    if snapshot.chunk_metrics.estimated_remaining_chunks is not None:
        remaining_chunks = snapshot.chunk_metrics.estimated_remaining_chunks
        if remaining_chunks >= 10:
            remaining_display = f"{remaining_chunks:,.0f}"
        else:
            remaining_display = f"{remaining_chunks:.1f}"
        llm_metrics.append(("Est. remaining chunks", remaining_display))
    llm_metrics.append(("Files sampled", str(snapshot.chunk_metrics.file_samples)))
    p75_display = (
        f"{snapshot.chunk_metrics.timeout_p75:.1f}s"
        if snapshot.chunk_metrics.timeout_p75 is not None
        else "—"
    )
    llm_metrics.append(("p75 chunk (s)", p75_display))
    if snapshot.chunk_metrics.last_file_chunks:
        last_avg = _format_seconds(snapshot.chunk_metrics.last_file_avg_seconds)
        llm_metrics.append(
            (
                "Last file",
                f"{snapshot.chunk_metrics.last_file_chunks} chunks @ {last_avg}",
            )
        )
    if snapshot.chunk_metrics.recent_chunk_counts:
        recent_counts = ", ".join(
            str(count) for count in snapshot.chunk_metrics.recent_chunk_counts
        )
        llm_metrics.append(("Recent chunk counts", recent_counts))
    llm_metrics.append(
        ("Last update", _format_time_since(snapshot.chunk_metrics.last_updated))
    )

    if len(llm_metrics) % 2:
        llm_metrics.append(("", ""))

    llm_table = Table.grid(expand=True, padding=(0, 3))
    llm_table.add_column(justify="left", ratio=1)
    llm_table.add_column(justify="left", ratio=1)
    for index in range(0, len(llm_metrics), 2):
        left_label, left_value = llm_metrics[index]
        right_label, right_value = llm_metrics[index + 1]
        left_cell = f"[bold]{left_label}:[/bold] {left_value}" if left_label else ""
        right_cell = f"[bold]{right_label}:[/bold] {right_value}" if right_label else ""
        llm_table.add_row(left_cell, right_cell)
    llm_panel = Panel(llm_table, title="LLM Throughput", border_style="magenta")

    queue_table = Table.grid(expand=True)
    queue_table.add_column(justify="left")
    queue_table.add_column(justify="right")
    for name, value in snapshot.queue_sizes.items():
        label = f"{name.capitalize()} queue"
        queue_table.add_row(label, "—" if value is None else str(value))
    queue_panel = Panel(queue_table, title="Queue Depths", border_style="green")

    worker_table = Table.grid(expand=True)
    worker_table.add_column(justify="left")
    worker_table.add_column(justify="right")
    for name, value in snapshot.worker_counts.items():
        worker_table.add_row(f"{name.capitalize()} workers", str(value))
    worker_panel = Panel(worker_table, title="Workers", border_style="cyan")

    status_grid = Table.grid(expand=True)
    status_grid.add_column(ratio=1)
    status_grid.add_column(ratio=1)
    status_grid.add_row(queue_panel, worker_panel)

    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    started_at = datetime.fromtimestamp(snapshot.start_timestamp).strftime("%H:%M:%S")
    header.add_row(
        f"[bold cyan]Model:[/bold cyan] {snapshot.model_label}",
        f"[bold cyan]Chunk limit:[/bold cyan] {snapshot.chunk_limit_label}",
    )
    header.add_row(
        f"[bold cyan]Started:[/bold cyan] {started_at}",
        f"[bold cyan]Active:[/bold cyan] {len(snapshot.active_files)} file(s)",
    )

    body = Group(
        header,
        Rule(style="dim"),
        progress_grid,
        Rule(style="dim"),
        llm_panel,
        status_grid,
    )

    return Panel(body, title="Pipeline Status", border_style="blue")


def _build_live_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=TOP_PANEL_ROWS),
        Layout(name="bottom"),
    )
    layout["top"].update(
        Panel("Initializing pipeline dashboard…", title="Pipeline Status")
    )
    initial_limit = _calculate_log_panel_display_limit(LIVE_CONSOLE)
    layout["bottom"].update(_render_logs_view([], total=0, display_limit=initial_limit))
    top_size = min(TOP_PANEL_ROWS, max(LIVE_CONSOLE.size.height - 1, 1))
    layout["top"].size = top_size
    return layout


def _update_progress_panel(layout: Layout, snapshot: ProgressSnapshot) -> None:
    """Update the top panel with the latest progress snapshot."""

    active_panel = _render_active_files_panel(snapshot.active_files)
    pipeline_panel = _render_progress_panel(snapshot)
    layout["top"].update(Group(active_panel, pipeline_panel))
    top_size = min(TOP_PANEL_ROWS, max(LIVE_CONSOLE.size.height - 1, 1))
    bottom_size = max(LIVE_CONSOLE.size.height - top_size, 0)
    layout["top"].size = top_size
    if bottom_size > 0:
        layout["bottom"].size = bottom_size


def _render_logs_view(lines: list[str], *, total: int, display_limit: int) -> Group:
    limit = max(display_limit, 1)
    visible = lines[-limit:] if lines else []
    display_count = len(visible)
    if total > display_count and display_count > 0:
        header_text = f"Logs (showing last {display_count} of {total})"
    else:
        header_text = "Logs"

    header = Text(header_text, style="bold")
    if visible:
        body_text = Text("\n".join(visible), overflow="crop", no_wrap=False)
    else:
        body_text = Text("No log messages yet.")
    aligned_body = Align(
        body_text,
        align="left",
        vertical="bottom",
        height=limit,
    )
    return Group(header, aligned_body)


class LiveLogHandler(logging.Handler):
    """Logging handler that streams records into the live Logs panel."""

    def __init__(
        self,
        layout: Layout,
        console: Console,
        *,
        panel_name: str = "bottom",
        max_lines: int | None = None,
        panel_ratio: float = LOG_PANEL_RATIO,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.console = console
        self.panel_name = panel_name
        self._panel_ratio = panel_ratio
        limit = max_lines if max_lines is not None and max_lines > 0 else None
        self._lines: deque[str]
        if limit is None:
            self._lines = deque()
        else:
            self._lines = deque(maxlen=limit)
        self._lock = threading.Lock()
        self._display_limit = _calculate_log_panel_display_limit(
            self.console, ratio=self._panel_ratio
        )
        self._last_console_height = self.console.size.height

    def _refresh_display_limit_if_needed(self) -> None:
        current_height = self.console.size.height
        if current_height == self._last_console_height:
            return
        self._display_limit = _calculate_log_panel_display_limit(
            self.console, ratio=self._panel_ratio
        )
        self._last_console_height = current_height

    def emit(self, record: logging.LogRecord) -> None:
        raw_message = record.getMessage()
        if raw_message:
            try:
                _apply_chunk_progress_from_log(raw_message)
            except Exception:  # pragma: no cover - defensive; avoid logging loops
                pass

        try:
            message = self.format(record)
        except Exception:
            message = raw_message or ""

        with self._lock:
            self._lines.append(message)
            lines_snapshot = list(self._lines)

        self._refresh_display_limit_if_needed()
        visible_lines = lines_snapshot[-self._display_limit :]

        self.layout[self.panel_name].update(
            _render_logs_view(
                visible_lines,
                total=len(lines_snapshot),
                display_limit=self._display_limit,
            )
        )


def signal_handler(signum, frame):
    """Handle termination signals by initiating a coordinated shutdown."""

    try:
        signal_name = signal.Signals(signum).name  # type: ignore[call-arg]
    except Exception:
        signal_name = str(signum)

    pipeline_logger.info("Received signal %s; initiating shutdown", signal_name)
    request_shutdown(f"signal {signal_name}", raise_interrupt=True)


def log_unprocessed_file(
    file_path: str, mime_type: str, reason: str, additional_info: str = ""
) -> None:
    """Log files that couldn't be processed for later analysis."""
    unprocessed_log_path = Path("data/unprocessed_files.csv")
    unprocessed_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists to determine if we need headers
    file_exists = unprocessed_log_path.exists()

    with lock:
        with unprocessed_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "file_path",
                        "mime_type",
                        "reason",
                        "additional_info",
                    ]
                )
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    file_path,
                    mime_type,
                    reason,
                    additional_info,
                ]
            )


def filter_records_by_search_term(
    records: list[FileRecord], search_term: str
) -> list[FileRecord]:
    """Return manifest records whose path or basename matches the search term.

    Performs case-insensitive substring matching on the full path and requires an
    exact, case-insensitive match on the filename to align with the CLI help text.
    """
    normalized_term = search_term.strip().lower()
    if not normalized_term:
        return []

    matches: list[FileRecord] = []
    for record in records:
        record_path_lower = record.file_path.lower()
        if normalized_term in record_path_lower:
            matches.append(record)
            continue

        if Path(record.file_path).name.lower() == normalized_term:
            matches.append(record)

    return matches


def has_text_content(file_record: FileRecord) -> bool:
    """Return True if file_record has non-empty extracted_text."""
    return bool((file_record.extracted_text or "").strip())


def _read_text_file_best_effort(file_path: str) -> str:
    """Read a text file, trying a few common encodings before giving up."""
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error: Exception | None = None

    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding) as handle:
                text = handle.read()
                if encoding != "utf-8":
                    extraction_logger.debug(
                        "Loaded %s using fallback encoding %s", file_path, encoding
                    )
                return text
        except UnicodeDecodeError as exc:
            extraction_logger.debug(
                "Failed to decode %s as %s: %s", file_path, encoding, exc
            )
            last_error = exc
        except Exception as exc:  # pragma: no cover - unexpected I/O failure
            extraction_logger.warning(
                "Unexpected error reading %s with encoding %s: %s",
                file_path,
                encoding,
                exc,
            )
            last_error = exc

    try:
        with open(file_path, "rb") as handle:
            raw_bytes = handle.read()
        text = raw_bytes.decode("utf-8", errors="ignore")
        if text.strip():
            extraction_logger.info(
                "Decoded %s with utf-8/ignore after failures; chars may be missing",
                file_path,
            )
            return text

        ascii_pattern = re.compile(
            rb"[ -~]{" + str(MIN_ASCII_CHUNK_LENGTH).encode("ascii") + rb",}"
        )
        ascii_chunks = [
            match.group().decode("ascii", errors="ignore")
            for match in ascii_pattern.finditer(raw_bytes)
        ]
        if ascii_chunks:
            extraction_logger.info(
                "Recovered %d ASCII chunk(s) from %s using strings-style fallback",
                len(ascii_chunks),
                file_path,
            )
            return "\n".join(ascii_chunks)
    except Exception as exc:  # pragma: no cover - unexpected I/O failure
        extraction_logger.error(
            "Failed to read text file %s after exhausting fallbacks: %s "
            "(last error: %s)",
            file_path,
            exc,
            last_error,
        )
        return ""


def extract_text_from_svg(file_path: str) -> str:
    """Extract visible text from an SVG, fallback to raw XML if parsing fails."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text_chunks: list[str] = []
        for element in root.iter():
            if element.text:
                chunk = element.text.strip()
                if chunk:
                    text_chunks.append(chunk)
        return "\n".join(text_chunks)
    except Exception as exc:
        extraction_logger.warning("Failed to parse SVG %s: %s", file_path, exc)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                return handle.read()
        except Exception as raw_exc:
            extraction_logger.error(
                "Failed to read SVG fallback %s: %s", file_path, raw_exc
            )
            return ""


def load_manifest() -> list[FileRecord]:
    """Load the manifest and parse it into a list of FileRecord objects."""
    if not MANIFEST_PATH.exists():
        return []

    def _load_from_path(path: Path) -> list[FileRecord]:
        with path.open() as handle:
            data = json.load(handle)
        return [FileRecord.model_validate(item) for item in data]

    try:
        return _load_from_path(MANIFEST_PATH)
    except json.JSONDecodeError as exc:
        pipeline_logger.error(
            "Manifest %s is corrupted: %s",
            MANIFEST_PATH,
            exc,
        )
        backups: list[tuple[float, Path]] = []
        if MANIFEST_BACKUP_DIR.exists():
            for candidate in MANIFEST_BACKUP_DIR.glob("manifest-*.json.bak"):
                try:
                    backups.append((candidate.stat().st_mtime, candidate))
                except OSError:
                    continue
            backups.sort(key=lambda item: item[0], reverse=True)

        for _, backup_path in backups:
            try:
                records = _load_from_path(backup_path)
                copy2(backup_path, MANIFEST_PATH)
                pipeline_logger.warning("Restored manifest from backup %s", backup_path)
                return records
            except json.JSONDecodeError:
                pipeline_logger.warning(
                    "Skipping corrupted manifest backup %s", backup_path
                )
                continue
            except Exception as backup_exc:  # pragma: no cover - rare filesystem issue
                pipeline_logger.warning(
                    "Unable to load manifest backup %s: %s", backup_path, backup_exc
                )
                continue

        raise RuntimeError(
            "Manifest file is corrupted and no valid backup was found."
        ) from exc


def _backup_manifest() -> None:
    """Create a timestamped backup of the manifest, trimming old copies."""

    if not MANIFEST_PATH.exists():
        return

    backup_logger = pipeline_logger.getChild("backup")
    try:
        MANIFEST_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        backup_logger.warning("Unable to create manifest backup directory: %s", exc)
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = MANIFEST_BACKUP_DIR / f"manifest-{timestamp}.json.bak"
    suffix = 1
    max_suffix_attempts = MAX_BACKUP_SUFFIX_ATTEMPTS
    while backup_path.exists():
        if suffix > max_suffix_attempts:
            backup_logger.warning(
                "Unable to determine unique manifest backup name after %d attempts",
                max_suffix_attempts,
            )
            return
        backup_path = (
            MANIFEST_BACKUP_DIR / f"manifest-{timestamp}-{suffix:02d}.json.bak"
        )
        suffix += 1
    try:
        copy2(MANIFEST_PATH, backup_path)
        backup_logger.debug("Created manifest backup at %s", backup_path)
    except Exception as exc:
        backup_logger.warning("Failed to create manifest backup: %s", exc)
        return

    try:
        backup_entries: list[tuple[float, Path]] = []
        for candidate in MANIFEST_BACKUP_DIR.glob("manifest-*.json.bak"):
            try:
                mtime = candidate.stat().st_mtime
            except OSError as exc:  # pragma: no cover  # unlikely edge case
                backup_logger.debug("Unable to stat backup %s: %s", candidate, exc)
                continue
            backup_entries.append((mtime, candidate))
    except Exception as exc:
        backup_logger.warning("Unable to enumerate manifest backups: %s", exc)
        return

    backup_entries.sort(key=lambda item: item[0], reverse=True)

    for _, stale in backup_entries[MANIFEST_BACKUP_LIMIT:]:
        try:
            stale.unlink()
            backup_logger.debug("Removed old manifest backup %s", stale)
        except Exception as exc:
            backup_logger.debug("Unable to remove backup %s: %s", stale, exc)


def save_manifest(manifest: list[FileRecord]) -> None:
    """Save the manifest to JSON file."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [record.model_dump(mode="json") for record in manifest]

    with manifest_write_lock:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=MANIFEST_PATH.parent,
                prefix=f"{MANIFEST_PATH.stem}-",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                temp_path = Path(tmp_file.name)
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(temp_path, MANIFEST_PATH)
            temp_path = None
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass


def extraction_worker(worker_id: int) -> None:
    """Worker thread for content extraction."""
    worker_logger = extraction_logger.getChild(f"worker-{worker_id}")
    worker_logger.debug("Extraction worker %d started", worker_id)

    while True:
        work_item = None
        try:
            work_item = extraction_queue.get(timeout=1.0)
            if work_item is None:  # Poison pill
                worker_logger.debug(
                    "Extraction worker %d received shutdown signal", worker_id
                )
                extraction_queue.task_done()
                break

            file_record = work_item.file_record
            correlation_id = work_item.correlation_id

            worker_logger.debug(
                "Worker %d extracting content from %s (MIME: %s) [%s]",
                worker_id,
                file_record.file_path,
                file_record.mime_type,
                correlation_id,
            )

            try:
                _check_for_shutdown()
                stage_start = time.time()
                extracted_frames: list[str] = []
                extracted_text = ""

                if (
                    file_record.mime_type.startswith("text/")
                    or file_record.mime_type == "application/pdf"
                ):
                    _check_for_shutdown()
                    if file_record.mime_type == "application/pdf":
                        extracted_text = extract_content_from_pdf(file_record.file_path)
                    else:
                        _check_for_shutdown()
                        extracted_text = _read_text_file_best_effort(
                            file_record.file_path
                        )
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type == (
                    "application/vnd.openxmlformats-"
                    "officedocument.wordprocessingml.document"
                ):
                    _check_for_shutdown()
                    extracted_text = extract_content_from_docx(file_record.file_path)
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type == (
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ):
                    _check_for_shutdown()
                    extracted_text = extract_content_from_xlsx(file_record.file_path)
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type == "image/svg+xml":
                    _check_for_shutdown()
                    extracted_text = extract_text_from_svg(file_record.file_path)
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type.startswith("image/"):
                    _check_for_shutdown()
                    extracted_text = extract_content_from_image(file_record.file_path)
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type.startswith(
                    "video/"
                ) and not file_record.file_path.lower().endswith(".asx"):
                    _check_for_shutdown()
                    extracted_frames = extract_frames_from_video(
                        file_record.file_path, "data/frames", interval_sec=10
                    )
                    file_record.extracted_frames = extracted_frames

                elif file_record.mime_type == "application/x-msaccess":
                    _check_for_shutdown()
                    ensure_required_tasks(file_record)

                _check_for_shutdown()
                extraction_duration = time.time() - stage_start
                extracted_text_len = len(file_record.extracted_text or "")
                extracted_frame_count = len(file_record.extracted_frames or [])

                worker_logger.info(
                    "Extraction complete for %s [%s] in %.2fs "
                    "(text=%d chars, frames=%d)",
                    file_record.file_path,
                    correlation_id,
                    extraction_duration,
                    extracted_text_len,
                    extracted_frame_count,
                )

                file_record.status = PENDING_ANALYSIS
                _check_for_shutdown()
                analysis_item = WorkItem(file_record, correlation_id)
                analysis_queue.put(analysis_item)

                worker_logger.debug(
                    "Worker %d completed extraction for %s [%s]",
                    worker_id,
                    file_record.file_path,
                    correlation_id,
                )

            except ShutdownRequested:
                worker_logger.info(
                    "Extraction cancelled for %s [%s] due to shutdown",
                    file_record.file_path,
                    correlation_id,
                )
                file_record.status = FAILED
                log_unprocessed_file(
                    file_record.file_path,
                    file_record.mime_type,
                    "shutdown_requested",
                    "extraction_cancelled",
                )
                with lock:
                    failed_files.add(correlation_id)
            except Exception as e:
                worker_logger.error(
                    "Worker %d failed to extract content from %s [%s]: %s",
                    worker_id,
                    file_record.file_path,
                    correlation_id,
                    e,
                )
                log_unprocessed_file(
                    file_record.file_path,
                    file_record.mime_type,
                    "extraction_failed",
                    str(e),
                )
                with lock:
                    failed_files.add(correlation_id)

            extraction_queue.task_done()

        except queue.Empty:
            if shutdown_event.is_set():
                break
            continue
        except Exception as e:
            worker_logger.error("Extraction worker %d error: %s", worker_id, e)

    worker_logger.debug("Extraction worker %d stopped", worker_id)


def analysis_worker(
    worker_id: int, model: AnalysisModel, max_chunks: int | None = None
) -> None:
    """Worker thread for AI analysis."""
    worker_logger = analysis_logger.getChild(f"worker-{worker_id}")
    worker_logger.debug("Analysis worker %d started", worker_id)

    while True:
        work_item = None
        correlation_id: str | None = None
        chunk_progress = FileChunkProgress()

        try:
            work_item = analysis_queue.get(timeout=1.0)
            if work_item is None:
                worker_logger.debug(
                    "Analysis worker %d received shutdown signal", worker_id
                )
                analysis_queue.task_done()
                break

            file_record = work_item.file_record
            correlation_id = work_item.correlation_id

            display_name = file_record.file_name or Path(file_record.file_path).name
            total_tasks = len(file_record.analysis_tasks)
            completed_tasks = sum(
                1
                for task in file_record.analysis_tasks
                if task.status == AnalysisStatus.COMPLETE
            )
            with lock:
                in_progress_files[correlation_id] = ActiveFileStatus(
                    file_name=display_name,
                    stage="Analyzing",
                    tasks_total=total_tasks,
                    tasks_completed=completed_tasks,
                )

            def _refresh_task_progress() -> None:
                completed = sum(
                    1
                    for task in file_record.analysis_tasks
                    if task.status == AnalysisStatus.COMPLETE
                )
                total = len(file_record.analysis_tasks)
                with lock:
                    status = in_progress_files.get(correlation_id)
                    if status:
                        status.tasks_completed = completed
                        status.tasks_total = total

            def _track_chunk_metrics(duration: float, chunk_count: int) -> None:
                if not chunk_progress.add(chunk_count=chunk_count, duration=duration):
                    return
                with lock:
                    status = in_progress_files.get(correlation_id)
                    if status:
                        status.chunks_processed = chunk_progress.count
                        status.chunks_total = max(
                            status.chunks_total or 0, chunk_progress.count
                        )

            worker_logger.debug(
                "Worker %d analyzing content from %s [%s]",
                worker_id,
                file_record.file_path,
                correlation_id,
            )
            _update_active_file_status(
                correlation_id, worker_logger, current_task="Preparing tasks"
            )
            _refresh_task_progress()

            handed_to_database = (
                False  # analysis worker cleans up only if enqueue fails
            )

            try:
                _check_for_shutdown()
                analysis_start = time.time()

                text_analysis_result: dict[str, Any] | None = None
                estate_analysis_result: dict[str, Any] | None = None
                source_name = file_record.file_name or Path(file_record.file_path).name

                for task in file_record.analysis_tasks:
                    _check_for_shutdown()
                    if task.status != AnalysisStatus.PENDING:
                        continue
                    if model != AnalysisModel.ALL and task.name.value != model.value:
                        continue

                    task_label = task.name.value.replace("_", " ").title()
                    _update_active_file_status(
                        correlation_id,
                        worker_logger,
                        stage="Analyzing",
                        current_task=task_label,
                    )

                    try:
                        if task.name in TEXT_BASED_ANALYSES:
                            if not has_text_content(file_record):
                                if task.name == AnalysisName.PASSWORD_DETECTION:
                                    file_record.contains_password = False
                                    file_record.passwords = {}
                                elif task.name == AnalysisName.ESTATE_ANALYSIS:
                                    file_record.has_estate_relevant_info = False
                                    file_record.estate_information = {}
                                worker_logger.debug(
                                    "Skipping %s for %s due to missing text",
                                    task.name.value,
                                    file_record.file_path,
                                )
                                task.status = AnalysisStatus.COMPLETE
                                _refresh_task_progress()
                                continue

                            if task.name == AnalysisName.PASSWORD_DETECTION:
                                _check_for_shutdown()
                                chunk_estimate = _estimate_chunk_count(
                                    file_record.extracted_text,
                                    max_chunks=max_chunks,
                                )
                                _increment_active_chunks_total(
                                    correlation_id, increment=chunk_estimate
                                )
                                operation_start = time.time()
                                password_result = detect_passwords(
                                    file_record.extracted_text,
                                    source_name=source_name,
                                    should_abort=_check_for_shutdown,
                                    max_chunks=max_chunks,
                                )
                                duration = time.time() - operation_start
                                actual_chunks = password_result.get("_chunk_count")
                                _track_chunk_metrics(
                                    duration,
                                    _resolve_chunk_metric(
                                        actual_chunks, chunk_estimate
                                    ),
                                )
                                file_record.contains_password = password_result.get(
                                    "contains_password"
                                )
                                file_record.passwords = password_result.get(
                                    "passwords", {}
                                )
                                if file_record.contains_password:
                                    worker_logger.info(
                                        "Password detector found %d candidate(s) in %s",
                                        len(file_record.passwords),
                                        file_record.file_path,
                                    )
                                else:
                                    worker_logger.debug(
                                        "Password detector found no passwords for %s",
                                        file_record.file_path,
                                    )
                                task.status = AnalysisStatus.COMPLETE
                                _refresh_task_progress()
                                continue
                            if task.name == AnalysisName.ESTATE_ANALYSIS:
                                if estate_analysis_result is None:
                                    _check_for_shutdown()
                                    chunk_estimate = _estimate_chunk_count(
                                        file_record.extracted_text,
                                        max_chunks=max_chunks,
                                    )
                                    _increment_active_chunks_total(
                                        correlation_id, increment=chunk_estimate
                                    )
                                    operation_start = time.time()
                                    estate_analysis_result = (
                                        analyze_estate_relevant_information(
                                            file_record.extracted_text,
                                            source_name=source_name,
                                            should_abort=_check_for_shutdown,
                                            max_chunks=max_chunks,
                                        )
                                    )
                                    duration = time.time() - operation_start
                                    actual_chunks = estate_analysis_result.get(
                                        "_chunk_count"
                                    )
                                    _track_chunk_metrics(
                                        duration,
                                        _resolve_chunk_metric(
                                            actual_chunks, chunk_estimate
                                        ),
                                    )
                                file_record.estate_information = (
                                    estate_analysis_result.get("estate_information", {})
                                )
                                has_flag = estate_analysis_result.get(
                                    "has_estate_relevant_info"
                                )
                                if has_flag is None:
                                    has_flag = bool(file_record.estate_information)
                                file_record.has_estate_relevant_info = bool(has_flag)
                                task.status = AnalysisStatus.COMPLETE
                                _refresh_task_progress()
                                continue

                            if text_analysis_result is None:
                                _check_for_shutdown()
                                chunk_estimate = _estimate_chunk_count(
                                    file_record.extracted_text,
                                    max_chunks=max_chunks,
                                )
                                _increment_active_chunks_total(
                                    correlation_id, increment=chunk_estimate
                                )
                                operation_start = time.time()
                                text_analysis_result = analyze_text_content(
                                    file_record.extracted_text,
                                    source_name=source_name,
                                    should_abort=_check_for_shutdown,
                                    max_chunks=max_chunks,
                                )
                                duration = time.time() - operation_start
                                actual_chunks = text_analysis_result.get("_chunk_count")
                                _track_chunk_metrics(
                                    duration,
                                    _resolve_chunk_metric(
                                        actual_chunks, chunk_estimate
                                    ),
                                )
                            if task.name == AnalysisName.TEXT_ANALYSIS:
                                file_record.summary = text_analysis_result.get(
                                    "summary"
                                )
                            elif task.name == AnalysisName.PEOPLE_ANALYSIS:
                                file_record.mentioned_people = text_analysis_result.get(
                                    "mentioned_people", []
                                )
                            task.status = AnalysisStatus.COMPLETE
                            _refresh_task_progress()
                            continue
                        elif task.name == AnalysisName.IMAGE_DESCRIPTION:
                            if file_record.mime_type.startswith("image/"):
                                _check_for_shutdown()
                                description = describe_image(
                                    file_record.file_path,
                                    should_abort=_check_for_shutdown,
                                )
                                file_record.description = description
                                if not file_record.summary:
                                    file_record.summary = description

                        elif task.name == AnalysisName.NSFW_CLASSIFICATION:
                            if file_record.mime_type.startswith("image/"):
                                _check_for_shutdown()
                                classifier = NSFWClassifier()
                                _check_for_shutdown()
                                nsfw_result = classifier.classify_image(
                                    file_record.file_path
                                )
                                file_record.is_nsfw = (
                                    nsfw_result["label"].lower() == "nsfw"
                                )
                            elif file_record.mime_type.startswith("video/"):
                                if file_record.extracted_frames:
                                    classifier = NSFWClassifier()
                                    for frame_path in file_record.extracted_frames:
                                        _check_for_shutdown()
                                        nsfw_result = classifier.classify_image(
                                            frame_path
                                        )
                                        if nsfw_result["label"].lower() == "nsfw":
                                            file_record.is_nsfw = True
                                            break

                        elif task.name == AnalysisName.VIDEO_SUMMARY:
                            if file_record.mime_type.startswith("video/") and not (
                                file_record.file_path.lower().endswith(".asx")
                            ):
                                _check_for_shutdown()
                                if not file_record.extracted_frames:
                                    worker_logger.warning(
                                        "No frames extracted for video %s, "
                                        "cannot generate summary",
                                        file_record.file_path,
                                    )
                                    log_unprocessed_file(
                                        file_record.file_path,
                                        file_record.mime_type,
                                        "no_video_frames_extracted",
                                        "Video too short or frame extraction failed",
                                    )
                                else:
                                    frame_descriptions = []
                                    for frame_path in file_record.extracted_frames:
                                        _check_for_shutdown()
                                        try:
                                            description = describe_image(
                                                frame_path,
                                                should_abort=_check_for_shutdown,
                                            )
                                            frame_descriptions.append(description)
                                        except Exception as e:
                                            worker_logger.warning(
                                                "Failed to describe frame %s: %s",
                                                frame_path,
                                                e,
                                            )
                                    if frame_descriptions:
                                        _check_for_shutdown()
                                        video_summary = summarize_video_frames(
                                            frame_descriptions,
                                            should_abort=_check_for_shutdown,
                                        )
                                        file_record.description = video_summary
                                        if not file_record.summary:
                                            file_record.summary = video_summary
                                        worker_logger.info(
                                            "Generated video summary for %s "
                                            "with %d frames",
                                            file_record.file_path,
                                            len(frame_descriptions),
                                        )
                                    else:
                                        worker_logger.warning(
                                            "No frame descriptions generated "
                                            "for video %s",
                                            file_record.file_path,
                                        )
                                        log_unprocessed_file(
                                            file_record.file_path,
                                            file_record.mime_type,
                                            "video_frame_description_failed",
                                            "All frame descriptions failed",
                                        )
                            else:
                                worker_logger.debug(
                                    "Skipping video summary for non-video file %s",
                                    file_record.file_path,
                                )

                        elif task.name == AnalysisName.FINANCIAL_ANALYSIS:
                            if file_record.extracted_text and (
                                "financial" in file_record.extracted_text.lower()
                                or "account" in file_record.extracted_text.lower()
                            ):
                                source_name = (
                                    file_record.file_name
                                    or Path(file_record.file_path).name
                                )
                                _check_for_shutdown()
                                chunk_estimate = _estimate_chunk_count(
                                    file_record.extracted_text,
                                    max_chunks=max_chunks,
                                )
                                _increment_active_chunks_total(
                                    correlation_id, increment=chunk_estimate
                                )
                                operation_start = time.time()
                                financial_analysis = analyze_financial_document(
                                    file_record.extracted_text,
                                    source_name=source_name,
                                    should_abort=_check_for_shutdown,
                                    max_chunks=max_chunks,
                                )
                                duration = time.time() - operation_start
                                actual_chunks = financial_analysis.get("_chunk_count")
                                _track_chunk_metrics(
                                    duration,
                                    _resolve_chunk_metric(
                                        actual_chunks, chunk_estimate
                                    ),
                                )
                                file_record.summary = financial_analysis.get("summary")
                                file_record.potential_red_flags = (
                                    financial_analysis.get("potential_red_flags")
                                )
                                file_record.incriminating_items = (
                                    financial_analysis.get("incriminating_items")
                                )
                                file_record.confidence_score = financial_analysis.get(
                                    "confidence_score"
                                )
                                file_record.has_financial_red_flags = bool(
                                    financial_analysis.get("potential_red_flags")
                                )

                        elif task.name == AnalysisName.ACCESS_DB_ANALYSIS:
                            _check_for_shutdown()
                            result = analyze_access_database(file_record.file_path)
                            if result.combined_text:
                                existing_text = file_record.extracted_text or ""
                                if existing_text:
                                    if result.combined_text not in existing_text:
                                        file_record.extracted_text = "\n\n".join(
                                            [existing_text, result.combined_text]
                                        )
                                else:
                                    file_record.extracted_text = result.combined_text
                            if result.text_analysis:
                                first_text_analysis = next(
                                    iter(result.text_analysis), None
                                )
                                if (
                                    first_text_analysis is not None
                                    and not file_record.summary
                                ):
                                    file_record.summary = first_text_analysis.summary
                            if result.financial_analysis:
                                worker_logger.info(
                                    "Financial analysis for %s: %s",
                                    file_record.file_path,
                                    result.financial_analysis,
                                )

                        task.status = AnalysisStatus.COMPLETE
                        _refresh_task_progress()

                    except ShutdownRequested:
                        raise
                    except Exception as e:
                        worker_logger.error(
                            "Failed to run analysis task %s for %s: %s",
                            task.name,
                            file_record.file_path,
                            e,
                        )
                        task.status = AnalysisStatus.ERROR
                        task.error_message = str(e)
                        _refresh_task_progress()

                _check_for_shutdown()
                database_item = WorkItem(file_record, correlation_id)
                database_queue.put(database_item)
                handed_to_database = True
                _update_active_file_status(
                    correlation_id,
                    worker_logger,
                    stage="Database",
                    current_task="Writing results",
                )

                completed_tasks = sum(
                    task.status == AnalysisStatus.COMPLETE
                    for task in file_record.analysis_tasks
                )
                failed_tasks = sum(
                    task.status == AnalysisStatus.ERROR
                    for task in file_record.analysis_tasks
                )
                pending_tasks = sum(
                    task.status == AnalysisStatus.PENDING
                    for task in file_record.analysis_tasks
                )
                analysis_duration = time.time() - analysis_start

                worker_logger.info(
                    "Analysis complete for %s [%s] in %.2fs "
                    "(tasks: %d complete, %d failed, %d pending, nsfw=%s, summary=%s)",
                    file_record.file_path,
                    correlation_id,
                    analysis_duration,
                    completed_tasks,
                    failed_tasks,
                    pending_tasks,
                    file_record.is_nsfw,
                    bool(file_record.summary),
                )

                worker_logger.debug(
                    "Worker %d completed analysis for %s [%s]",
                    worker_id,
                    file_record.file_path,
                    correlation_id,
                )
            except ShutdownRequested:
                worker_logger.info(
                    "Analysis cancelled for %s [%s] due to shutdown",
                    file_record.file_path,
                    correlation_id,
                )
                _update_active_file_status(
                    correlation_id, worker_logger, stage="Cancelled", current_task=None
                )
                file_record.status = FAILED
                log_unprocessed_file(
                    file_record.file_path,
                    file_record.mime_type,
                    "shutdown_requested",
                    "analysis_cancelled",
                )
                with lock:
                    failed_files.add(correlation_id)
            except Exception as e:
                worker_logger.error(
                    "Worker %d failed to analyze content from %s [%s]: %s",
                    worker_id,
                    file_record.file_path,
                    correlation_id,
                    e,
                )
                _update_active_file_status(
                    correlation_id, worker_logger, stage="Error", current_task=None
                )
                log_unprocessed_file(
                    file_record.file_path,
                    file_record.mime_type,
                    "analysis_failed",
                    str(e),
                )
                with lock:
                    failed_files.add(correlation_id)
            finally:
                if chunk_progress.duration >= 0 and chunk_progress.count > 0:
                    _record_file_chunk_metrics(
                        chunk_progress.duration, chunk_progress.count
                    )
                if correlation_id is not None and not handed_to_database:
                    # For items that never made it to the database queue, ensure
                    # we release the active-file entry here; otherwise the
                    # database worker removes it after indexing (see database worker).
                    with lock:
                        in_progress_files.pop(correlation_id, None)

            analysis_queue.task_done()

        except queue.Empty:
            if shutdown_event.is_set():
                break
            continue
        except Exception as e:
            worker_logger.error("Analysis worker %d error: %s", worker_id, e)

    worker_logger.debug("Analysis worker %d stopped", worker_id)


def database_worker(worker_id: int, collection: Any) -> None:
    """Worker thread for database operations."""
    worker_logger = database_logger.getChild(f"worker-{worker_id}")
    worker_logger.debug("Database worker %d started", worker_id)

    while True:
        work_item = None
        try:
            work_item = database_queue.get(timeout=1.0)
            if work_item is None:
                worker_logger.debug(
                    "Database worker %d received shutdown signal", worker_id
                )
                database_queue.task_done()
                break

            file_record = work_item.file_record
            correlation_id = work_item.correlation_id

            worker_logger.debug(
                "Worker %d adding to database: %s [%s]",
                worker_id,
                file_record.file_path,
                correlation_id,
            )
            with lock:
                status = in_progress_files.get(correlation_id)
                if status:
                    status.stage = "Database"
                    status.current_task = "Indexing"

            try:
                _check_for_shutdown()
                add_file_to_db(file_record.model_dump(), collection)

                all_tasks_done = all(
                    task.status != AnalysisStatus.PENDING
                    for task in file_record.analysis_tasks
                )
                any_task_failed = any(
                    task.status == AnalysisStatus.ERROR
                    for task in file_record.analysis_tasks
                )

                target_set = completed_files
                if not all_tasks_done:
                    file_record.status = FAILED
                    target_set = failed_files

                    pending_tasks = ", ".join(
                        task.name.value
                        for task in file_record.analysis_tasks
                        if task.status == AnalysisStatus.PENDING
                    )
                    log_unprocessed_file(
                        file_record.file_path,
                        file_record.mime_type,
                        "analysis_tasks_incomplete",
                        pending_tasks or "Analysis tasks never completed",
                    )
                elif any_task_failed:
                    file_record.status = FAILED
                    target_set = failed_files

                    failed_details = ", ".join(
                        f"{task.name.value}: {task.error_message or 'unknown error'}"
                        for task in file_record.analysis_tasks
                        if task.status == AnalysisStatus.ERROR
                    )
                    log_unprocessed_file(
                        file_record.file_path,
                        file_record.mime_type,
                        "analysis_task_failed",
                        failed_details or "One or more analysis tasks failed",
                    )
                else:
                    file_record.status = COMPLETE

                with lock:
                    target_set.add(correlation_id)

                worker_logger.info(
                    "Database indexed %s [%s] with status=%s "
                    "(nsfw=%s, financial_flags=%s)",
                    file_record.file_path,
                    correlation_id,
                    file_record.status,
                    file_record.is_nsfw,
                    file_record.has_financial_red_flags,
                )

                worker_logger.debug(
                    "Worker %d completed database addition for %s [%s]",
                    worker_id,
                    file_record.file_path,
                    correlation_id,
                )

            except ShutdownRequested:
                worker_logger.info(
                    "Database update cancelled for %s [%s] due to shutdown",
                    file_record.file_path,
                    correlation_id,
                )
                file_record.status = FAILED
                log_unprocessed_file(
                    file_record.file_path,
                    file_record.mime_type,
                    "shutdown_requested",
                    "database_cancelled",
                )
                with lock:
                    failed_files.add(correlation_id)
            except Exception as e:
                worker_logger.error(
                    "Worker %d failed to add %s to database [%s]: %s",
                    worker_id,
                    file_record.file_path,
                    correlation_id,
                    e,
                )
                with lock:
                    failed_files.add(correlation_id)
            finally:
                with lock:
                    # Analysis worker clears entries if enqueue fails; on success
                    # the database worker owns final cleanup here.
                    in_progress_files.pop(correlation_id, None)

            database_queue.task_done()

        except queue.Empty:
            if shutdown_event.is_set():
                break
            continue
        except Exception as e:
            worker_logger.error("Database worker %d error: %s", worker_id, e)

    worker_logger.debug("Database worker %d stopped", worker_id)


def save_manifest_periodically(manifest: list[FileRecord]) -> None:
    """Periodically save manifest to persist progress."""
    thread_logger = pipeline_logger.getChild("manifest_saver")
    while True:
        time.sleep(10)
        try:
            save_manifest(manifest)
            thread_logger.debug("Manifest saved periodically")
        except Exception as e:
            thread_logger.warning("Failed to save manifest: %s", e)


def write_processed_files_to_csv(processed_files: list[dict], file_path: str) -> None:
    """Write processed file data to a CSV file using pandas."""
    if not processed_files:
        return

    df = pd.DataFrame(processed_files)
    df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)


def main(
    model: Annotated[
        AnalysisModel, typer.Option(help="The model to use for analysis.")
    ] = AnalysisModel.ALL,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Number of files to process in this batch (0 = process all remaining)."
        ),
    ] = 0,
    csv_output: Annotated[
        str,
        typer.Option(
            help="Path to save a CSV file of processed files.", rich_help_panel="Output"
        ),
    ] = "",
    target_filename: Annotated[
        str,
        typer.Option(
            "--file-name",
            help="Only process files whose path or basename contains this value.",
            rich_help_panel="Filtering",
        ),
    ] = "",
    per_mime_limit: Annotated[
        int,
        typer.Option(
            help=(
                "Maximum number of files per MIME type to process in this run "
                "(0 disables the limit)."
            ),
            rich_help_panel="Filtering",
        ),
    ] = 0,
    max_chunks: Annotated[
        int,
        typer.Option(
            help=(
                "Maximum number of text chunks to analyze per document "
                "(0 disables the limit)."
            ),
            rich_help_panel="Processing",
        ),
    ] = 0,
    max_threads: Annotated[
        int,
        typer.Option(
            help=(
                "Maximum number of worker threads across extraction, analysis, "
                "and database stages (0 uses defaults)."
            ),
            rich_help_panel="Processing",
        ),
    ] = 0,
    debug: Annotated[bool, typer.Option(help="Enable debug logging.")] = False,
) -> int:
    """Run the main threaded pipeline."""
    console_logging = target_filename != "" or debug

    configure_logging(
        level=logging.DEBUG if debug else logging.INFO,
        console=console_logging,
        force=True,
    )

    shutdown_event.clear()
    _shutdown_signals_sent.clear()
    completed_files.clear()
    failed_files.clear()
    in_progress_files.clear()

    import warnings

    warnings.filterwarnings("ignore", message=".*use_fast.*")
    warnings.filterwarnings("ignore", message=".*slow processor.*")
    warnings.filterwarnings("ignore", message=".*Device set to use.*")

    MUPDF_SCREEN_ERR_SUBSTR = (
        "MuPDF error: unsupported error: cannot create appearance stream for Screen"
    )

    class MuPDFErrorFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr

        def write(self, data):
            if MUPDF_SCREEN_ERR_SUBSTR not in data:
                self.original_stderr.write(data)

        def flush(self):
            self.original_stderr.flush()

    sys.stderr = MuPDFErrorFilter(sys.stderr)
    run_logger = pipeline_logger.getChild("run")
    run_logger.info("Starting threaded file catalog pipeline...")
    _backup_manifest()

    (
        extraction_workers,
        analysis_workers,
        database_workers,
        applied_thread_limit,
        limit_was_increased,
    ) = _resolve_worker_counts(max_threads)
    global _active_worker_counts
    _active_worker_counts = {
        "extraction": extraction_workers,
        "analysis": analysis_workers,
        "database": database_workers,
    }
    total_workers = extraction_workers + analysis_workers + database_workers
    default_total_workers = (
        NUM_EXTRACTION_WORKERS + NUM_ANALYSIS_WORKERS + NUM_DATABASE_WORKERS
    )
    if max_threads > 0:
        if limit_was_increased:
            run_logger.warning(
                "Requested max threads %d is below minimum %d; using %d instead.",
                max_threads,
                MINIMUM_WORKER_TOTAL,
                applied_thread_limit,
            )
        run_logger.info(
            "Worker threads capped at %d: extraction=%d, analysis=%d, database=%d "
            "(total=%d).",
            applied_thread_limit,
            extraction_workers,
            analysis_workers,
            database_workers,
            total_workers,
        )
    else:
        run_logger.info(
            "Worker threads default to extraction=%d, analysis=%d, database=%d "
            "(total=%d).",
            extraction_workers,
            analysis_workers,
            database_workers,
            total_workers,
        )
        if total_workers != default_total_workers:
            run_logger.debug(
                "Worker counts differ from defaults after initialization "
                "(defaults total %d).",
                default_total_workers,
            )

    max_chunk_limit = max_chunks if max_chunks > 0 else None
    if max_chunk_limit:
        run_logger.info(
            "Limiting text analyses to the first %d chunk(s) per document.",
            max_chunk_limit,
        )

    collection = initialize_db(str(DB_PATH))
    run_logger.debug("Database initialized")

    try:
        full_manifest = load_manifest()
    except RuntimeError as exc:
        message = str(exc)
        run_logger.error(message)
        typer.echo(message)
        return 1
    run_logger.info("Loaded %d files from manifest", len(full_manifest))

    reset_count = reset_outdated_analysis_tasks(full_manifest)
    if reset_count:
        run_logger.info("Reset %d analysis tasks due to version changes", reset_count)

    # Set up signal handlers for safe shutdown
    global current_manifest
    current_manifest = full_manifest
    run_logger.debug("Manifest reference stored for shutdown handling")

    candidate_manifest = [f for f in full_manifest if f.status != COMPLETE]
    if target_filename:
        filtered_manifest = filter_records_by_search_term(
            candidate_manifest, target_filename
        )
        fallback_used = False
        if not filtered_manifest:
            filtered_manifest = filter_records_by_search_term(
                full_manifest, target_filename
            )
            fallback_used = bool(filtered_manifest)
        if not filtered_manifest:
            message = (
                f"No files matched the filter {target_filename!r}. Nothing to process."
            )
            run_logger.warning(message)
            typer.echo(message)
            return 0

        if fallback_used:
            run_logger.info(
                "Targeted filter %r matched only completed files; "
                "reprocessing %d file(s).",
                target_filename,
                len(filtered_manifest),
            )

        run_logger.info(
            "Targeted run enabled; reprocessing %d file(s) for %r.",
            len(filtered_manifest),
            target_filename,
        )
        reprocess_message = (
            f"Reprocessing {len(filtered_manifest)} file(s) matching "
            f"{target_filename!r}."
        )
        typer.echo(reprocess_message)

        for record in filtered_manifest:
            reset_file_record_for_rescan(record)
            try:
                with lock:
                    collection.delete(ids=[record.file_path])
                run_logger.debug(
                    "Cleared previous database entry for %s", record.file_path
                )
            except Exception as exc:  # pragma: no cover - depends on chromadb impl
                run_logger.debug(
                    "No existing database entry to clear for %s: %s",
                    record.file_path,
                    exc,
                )

        processing_manifest = filtered_manifest
        run_logger.info(
            "Filtered manifest to %d file(s) matching %r and reset prior analysis.",
            len(processing_manifest),
            target_filename,
        )
    else:
        run_logger.info(
            "Found %d unprocessed files.",
            len(candidate_manifest),
        )
        processing_manifest = candidate_manifest

    if not processing_manifest:
        message = "No files selected for processing after filtering."
        run_logger.info(message)
        typer.echo(message)
        return 0

    if not target_filename and batch_size > 0:
        processing_manifest = candidate_manifest[:batch_size]
        run_logger.info(
            "Processing batch of %d file(s) from %d available.",
            len(processing_manifest),
            len(candidate_manifest),
        )
    else:
        run_logger.info(
            "Processing all remaining files (%d).", len(processing_manifest)
        )

    if batch_size > 0 and target_filename:
        limited_manifest = processing_manifest[:batch_size]
        if not limited_manifest:
            run_logger.warning(
                "Batch size %d left no files to process after filtering.",
                batch_size,
            )
            return 0
        if len(limited_manifest) < len(processing_manifest):
            run_logger.info(
                "Batch size limit reduced targeted set to %d file(s) (from %d).",
                len(limited_manifest),
                len(processing_manifest),
            )
        processing_manifest = limited_manifest

    if per_mime_limit > 0:
        mime_counts: dict[str, int] = {}
        limited_manifest: list[FileRecord] = []
        for record in processing_manifest:
            mime_key = record.mime_type or "unknown"
            count = mime_counts.get(mime_key, 0)
            if count >= per_mime_limit:
                continue
            mime_counts[mime_key] = count + 1
            limited_manifest.append(record)

        if not limited_manifest:
            run_logger.warning(
                "No files remain after applying per-MIME limit of %d.",
                per_mime_limit,
            )
            return 0

        if len(limited_manifest) < len(processing_manifest):
            run_logger.info(
                "Per-MIME limit reduced processing set to %d file(s) "
                "across %d MIME type(s).",
                len(limited_manifest),
                len(mime_counts),
            )

        processing_manifest = limited_manifest

    manifest_saver = threading.Thread(
        target=save_manifest_periodically, args=(full_manifest,), daemon=True
    )
    manifest_saver.start()
    run_logger.debug("Started manifest saver thread")

    threads = []
    for i in range(extraction_workers):
        thread = threading.Thread(
            target=extraction_worker, args=(i,), name=f"ExtractionWorker-{i}"
        )
        thread.start()
        threads.append(thread)
        extraction_logger.debug("Started extraction worker %d", i)

    for i in range(analysis_workers):
        thread = threading.Thread(
            target=analysis_worker,
            args=(i, model, max_chunk_limit),
            name=f"AnalysisWorker-{i}",
        )
        thread.start()
        threads.append(thread)
        analysis_logger.debug("Started analysis worker %d", i)

    for i in range(database_workers):
        thread = threading.Thread(
            target=database_worker, args=(i, collection), name=f"DatabaseWorker-{i}"
        )
        thread.start()
        threads.append(thread)
        database_logger.debug("Started database worker %d", i)

    run_logger.info("All worker threads started")

    pending_files = 0
    for file_record in processing_manifest:
        correlation_id = f"file-{hash(file_record.file_path) % 100000:05d}"

        run_logger.debug(
            "Queuing %s (status: %s) [%s]",
            file_record.file_path,
            file_record.status,
            correlation_id,
        )

        # Filter files based on model type before queuing
        should_process = True
        if model is AnalysisModel.VIDEO_ANALYZER:
            should_process = file_record.mime_type.startswith(
                "video/"
            ) and not file_record.file_path.lower().endswith(".asx")
        elif model is AnalysisModel.IMAGE_DESCRIBER:
            should_process = file_record.mime_type.startswith("image/")
        elif model.value in TEXT_BASED_ANALYSIS_MODEL_VALUES:
            should_process = (
                file_record.mime_type.startswith("text/")
                or file_record.mime_type == "application/pdf"
                or file_record.mime_type == "image/svg+xml"
                or file_record.mime_type.endswith("document")
                or file_record.mime_type.endswith("sheet")
            )

        if not should_process:
            run_logger.debug(
                "Skipping %s - not applicable for model %s [%s]",
                file_record.file_path,
                model.value,
                correlation_id,
            )
            continue

        if file_record.status == PENDING_EXTRACTION:
            work_item = WorkItem(file_record, correlation_id)
            extraction_queue.put(work_item)
            extraction_logger.info(
                "Queued %s for extraction [%s]", file_record.file_path, correlation_id
            )
            pending_files += 1
        elif file_record.status == PENDING_ANALYSIS:
            work_item = WorkItem(file_record, correlation_id)
            analysis_queue.put(work_item)
            analysis_logger.info(
                "Queued %s for analysis [%s]", file_record.file_path, correlation_id
            )
            pending_files += 1
        elif file_record.status == COMPLETE:
            run_logger.debug(
                "Skipping completed file %s [%s]",
                file_record.file_path,
                correlation_id,
            )
            with lock:
                completed_files.add(correlation_id)
        elif file_record.status == FAILED:
            run_logger.debug(
                "Skipping failed file %s [%s]",
                file_record.file_path,
                correlation_id,
            )
            with lock:
                failed_files.add(correlation_id)

    run_logger.info("Queued %d pending files for processing", pending_files)
    if pending_files == 0:
        message = (
            "No files were queued for processing. "
            "Verify that the filter matches pending content or that the file type "
            "is supported for the selected model."
        )
        run_logger.warning(message)
        typer.echo(message)

    layout = _build_live_layout()
    start_time = time.time()
    initial_snapshot = _collect_progress_snapshot(
        total=pending_files,
        completed=len(completed_files),
        failed=len(failed_files),
        remaining=pending_files,
        active=[],
        start_time=start_time,
        model=model,
        chunk_limit=max_chunk_limit,
    )
    _update_progress_panel(layout, initial_snapshot)

    root_logger = logging.getLogger()
    log_handler = LiveLogHandler(layout, LIVE_CONSOLE)
    log_handler.setLevel(logging.NOTSET)

    existing_formatter = next(
        (handler.formatter for handler in root_logger.handlers if handler.formatter),
        None,
    )

    if existing_formatter is not None:
        log_handler.setFormatter(existing_formatter)
    else:
        log_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    root_logger.addHandler(log_handler)

    last_progress_time = start_time
    try:

        with Live(layout, console=LIVE_CONSOLE, refresh_per_second=4):
            with lock:
                completed_count = len(completed_files)
                failed_count = len(failed_files)
                active_files = list(in_progress_files.values())

            initial_remaining = pending_files - (completed_count + failed_count)
            snapshot = _collect_progress_snapshot(
                total=pending_files,
                completed=completed_count,
                failed=failed_count,
                remaining=initial_remaining,
                active=active_files,
                start_time=start_time,
                model=model,
                chunk_limit=max_chunk_limit,
            )
            _update_progress_panel(layout, snapshot)

            while pending_files > 0 and not shutdown_event.is_set():
                with lock:
                    completed_count = len(completed_files)
                    failed_count = len(failed_files)
                    active_files = list(in_progress_files.values())

                processed_count = completed_count + failed_count
                remaining = pending_files - processed_count

                snapshot = _collect_progress_snapshot(
                    total=pending_files,
                    completed=completed_count,
                    failed=failed_count,
                    remaining=remaining,
                    active=active_files,
                    start_time=start_time,
                    model=model,
                    chunk_limit=max_chunk_limit,
                )
                _update_progress_panel(layout, snapshot)

                if shutdown_event.is_set():
                    break

                current_time = time.time()
                if current_time - last_progress_time >= 30:
                    elapsed = current_time - start_time
                    if processed_count > 0:
                        estimated_total = elapsed * pending_files / processed_count
                        estimated_remaining = estimated_total - elapsed
                        if active_files:
                            run_logger.info(
                                (
                                    "Progress: %d/%d completed, %d failed, %d "
                                    "remaining (%.1f%% complete, ~%.1f minutes "
                                    "remaining) | active: %s"
                                ),
                                completed_count,
                                pending_files,
                                failed_count,
                                remaining,
                                (processed_count / pending_files) * 100,
                                estimated_remaining / 60,
                                _format_active_files(active_files),
                            )
                        else:
                            run_logger.info(
                                (
                                    "Progress: %d/%d completed, %d failed, %d "
                                    "remaining (%.1f%% complete, ~%.1f minutes "
                                    "remaining)"
                                ),
                                completed_count,
                                pending_files,
                                failed_count,
                                remaining,
                                (processed_count / pending_files) * 100,
                                estimated_remaining / 60,
                            )
                    else:
                        if active_files:
                            run_logger.info(
                                (
                                    "Progress: %d/%d completed, %d failed, %d "
                                    "remaining | active: %s"
                                ),
                                completed_count,
                                pending_files,
                                failed_count,
                                remaining,
                                _format_active_files(active_files),
                            )
                        else:
                            run_logger.info(
                                "Progress: %d/%d completed, %d failed, %d remaining",
                                completed_count,
                                pending_files,
                                failed_count,
                                remaining,
                            )
                    last_progress_time = current_time

                if remaining <= 0:
                    break

                time.sleep(2)
                if shutdown_event.is_set():
                    break

            with lock:
                final_completed = len(completed_files)
                final_failed = len(failed_files)
                remaining_after = max(
                    pending_files - (final_completed + final_failed), 0
                )

            final_snapshot = _collect_progress_snapshot(
                total=pending_files,
                completed=final_completed,
                failed=final_failed,
                remaining=remaining_after,
                active=[],
                start_time=start_time,
                model=model,
                chunk_limit=max_chunk_limit,
            )
            _update_progress_panel(layout, final_snapshot)

    except KeyboardInterrupt:
        run_logger.info("Keyboard interrupt received; cancelling current work...")
        request_shutdown("keyboard_interrupt")
    finally:
        # Ensure manifest is saved even if there's an unexpected exception
        run_logger.info("Ensuring manifest is saved...")
        try:
            save_manifest(full_manifest)
            run_logger.info("Manifest saved successfully")
        except Exception as e:
            run_logger.error("Failed to save manifest: %s", e)
        if log_handler in root_logger.handlers:
            root_logger.removeHandler(log_handler)

    if shutdown_event.is_set():
        run_logger.info(
            "Shutdown requested before completion; skipping remaining queue drains."
        )
        lingering_threads = _join_threads_with_timeout(
            threads, timeout=SHUTDOWN_JOIN_TIMEOUT
        )
        if lingering_threads:
            run_logger.warning(
                "Shutdown timeout reached; threads still running: %s",
                ", ".join(lingering_threads),
            )
        save_manifest(full_manifest)
    else:
        run_logger.info("Waiting for all work to complete...")
        extraction_queue.join()
        analysis_queue.join()
        database_queue.join()

        run_logger.debug("Sending shutdown signals to workers...")
        dispatch_shutdown_to_workers()

        run_logger.debug("Waiting for worker threads to stop...")
        lingering_threads = _join_threads_with_timeout(
            threads, timeout=NORMAL_JOIN_TIMEOUT
        )
        if lingering_threads:
            run_logger.warning(
                "Threads still running after shutdown: %s",
                ", ".join(lingering_threads),
            )

        save_manifest(full_manifest)

    elapsed_time = time.time() - start_time
    with lock:
        completed_count = len(completed_files)
        failed_count = len(failed_files)

    if shutdown_event.is_set():
        run_logger.info(
            "Pipeline interrupted after %.1f seconds (%d completed, %d failed)",
            elapsed_time,
            completed_count,
            failed_count,
        )
    else:
        run_logger.info(
            "Pipeline complete! Processed %d files in %.1f seconds "
            "(%d completed, %d failed)",
            len(full_manifest),
            elapsed_time,
            completed_count,
            failed_count,
        )

    # Log files with incomplete analysis for future AI analyzer development
    incomplete_files = []
    for record in full_manifest:
        if record.status in {COMPLETE, FAILED}:
            incomplete_tasks = [
                task
                for task in record.analysis_tasks
                if task.status != AnalysisStatus.COMPLETE
            ]
            if incomplete_tasks:
                incomplete_files.append((record, incomplete_tasks))
                for task in incomplete_tasks:
                    log_unprocessed_file(
                        record.file_path,
                        record.mime_type,
                        "incomplete analysis",
                        f"tasks: {', '.join(t.name for t in incomplete_tasks)}",
                    )
    if csv_output:
        if shutdown_event.is_set():
            run_logger.info(
                "Skipping CSV export because the run was interrupted before completion."
            )
        else:
            processed_files_data = [
                record.model_dump()
                for record in processing_manifest
                if f"file-{hash(record.file_path) % 100000:05d}" in completed_files
            ]
            if processed_files_data:
                run_logger.info(
                    "Writing %d processed files to %s",
                    len(processed_files_data),
                    csv_output,
                )
                write_processed_files_to_csv(processed_files_data, csv_output)
            else:
                run_logger.info("No files to write to CSV.")

    if shutdown_event.is_set():
        run_logger.warning("Shutdown request interrupted processing before completion.")
        return 1

    if failed_count > 0:
        run_logger.warning("Some files failed to process. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    typer.run(main)
