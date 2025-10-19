#!/usr/bin/env python3
"""Main orchestrator for the file catalog pipeline."""

from __future__ import annotations

import os

# Disable tokenizers parallelism to avoid warnings in threaded environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import logging
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
import typer
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from typing_extensions import Annotated

from src.access_analysis import analyze_access_database
from src.ai_analyzer import (
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


# Constants
MANIFEST_PATH = Path("data/manifest.json")
DB_PATH = Path("data/chromadb")

TEXT_BASED_ANALYSES = {
    AnalysisName.TEXT_ANALYSIS,
    AnalysisName.PEOPLE_ANALYSIS,
    AnalysisName.PASSWORD_DETECTION,
}
TEXT_BASED_ANALYSIS_MODEL_VALUES = {analysis.value for analysis in TEXT_BASED_ANALYSES}


# Threading configuration
NUM_EXTRACTION_WORKERS = 2
NUM_ANALYSIS_WORKERS = 4  # More workers since AI analysis is the bottleneck
NUM_DATABASE_WORKERS = 1

# Shared state
extraction_queue = queue.Queue()
analysis_queue = queue.Queue()
database_queue = queue.Queue()
completed_files = set()
failed_files = set()
in_progress_files: dict[str, str] = {}
lock = threading.Lock()
shutdown_event = threading.Event()
_shutdown_signals_sent = threading.Event()

# Global manifest for signal handlers
current_manifest = None

LOGGER_NAME = "file_catalog.pipeline"
pipeline_logger = logging.getLogger(LOGGER_NAME)
extraction_logger = logging.getLogger(f"{LOGGER_NAME}.extraction")
analysis_logger = logging.getLogger(f"{LOGGER_NAME}.analysis")
database_logger = logging.getLogger(f"{LOGGER_NAME}.database")

LIVE_CONSOLE = Console()
LOG_PANEL_RATIO = 0.5


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


def _format_active_files(active: list[str], limit: int = 3) -> str:
    """Build a short preview of active files for logging."""

    if not active:
        return ""
    preview = ", ".join(active[:limit])
    return preview + ("..." if len(active) > limit else "")


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
        (NUM_EXTRACTION_WORKERS, extraction_queue),
        (NUM_ANALYSIS_WORKERS, analysis_queue),
        (NUM_DATABASE_WORKERS, database_queue),
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


def _build_live_layout() -> Layout:
    layout = Layout()
    bottom_units = max(int(round(LOG_PANEL_RATIO * 100)), 1)
    top_units = max(100 - bottom_units, 1)
    layout.split_column(
        Layout(name="top", ratio=top_units),
        Layout(name="bottom", ratio=bottom_units),
    )
    layout["top"].update(Panel("Initializing pipeline...", title="Progress"))
    initial_limit = _calculate_log_panel_display_limit(LIVE_CONSOLE)
    layout["bottom"].update(_render_logs_view([], total=0, display_limit=initial_limit))
    return layout


def _update_progress_panel(
    layout: Layout,
    *,
    completed: int,
    failed: int,
    remaining: int,
    active: list[str],
) -> None:
    remaining_value = max(remaining, 0)
    lines = [
        f"Completed: {completed}",
        f"Failed: {failed}",
        f"Remaining: {remaining_value}",
    ]

    if active:
        max_preview = 3
        lines.append("")
        lines.append("Active files:")
        lines.extend(active[:max_preview])
        if len(active) > max_preview:
            lines.append("...")

    top_body = "\n".join(lines)
    layout["top"].update(Panel(top_body, title="Progress"))


def _render_logs_view(lines: list[str], *, total: int, display_limit: int) -> Group:
    visible = lines if lines else []
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
        height=max(display_limit, 1),
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
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()

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
    with MANIFEST_PATH.open() as f:
        data = json.load(f)
        return [FileRecord.model_validate(item) for item in data]


def save_manifest(manifest: list[FileRecord]) -> None:
    """Save the manifest to JSON file."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w") as f:
        data = [record.model_dump(mode="json") for record in manifest]
        json.dump(data, f, indent=2)


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
                        try:
                            _check_for_shutdown()
                            with open(
                                file_record.file_path, "r", encoding="utf-8"
                            ) as f:
                                extracted_text = f.read()
                        except Exception:
                            extracted_text = ""
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


def analysis_worker(worker_id: int, model: AnalysisModel) -> None:
    """Worker thread for AI analysis."""
    worker_logger = analysis_logger.getChild(f"worker-{worker_id}")
    worker_logger.debug("Analysis worker %d started", worker_id)

    while True:
        work_item = None
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

            with lock:
                in_progress_files[correlation_id] = (
                    file_record.file_name or Path(file_record.file_path).name
                )

            worker_logger.debug(
                "Worker %d analyzing content from %s [%s]",
                worker_id,
                file_record.file_path,
                correlation_id,
            )

            try:
                _check_for_shutdown()
                stage_start = time.time()

                text_analysis_result: dict[str, Any] | None = None
                source_name = file_record.file_name or Path(file_record.file_path).name

                for task in file_record.analysis_tasks:
                    _check_for_shutdown()
                    if task.status != AnalysisStatus.PENDING:
                        continue
                    if model != AnalysisModel.ALL and task.name.value != model.value:
                        continue

                    try:
                        if task.name in TEXT_BASED_ANALYSES:
                            if not has_text_content(file_record):
                                if task.name == AnalysisName.PASSWORD_DETECTION:
                                    file_record.contains_password = False
                                    file_record.passwords = {}
                                worker_logger.debug(
                                    "Skipping %s for %s due to missing text",
                                    task.name.value,
                                    file_record.file_path,
                                )
                                task.status = AnalysisStatus.COMPLETE
                                continue

                            if task.name == AnalysisName.PASSWORD_DETECTION:
                                _check_for_shutdown()
                                password_result = detect_passwords(
                                    file_record.extracted_text,
                                    source_name=source_name,
                                    should_abort=_check_for_shutdown,
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
                                continue

                            if text_analysis_result is None:
                                _check_for_shutdown()
                                text_analysis_result = analyze_text_content(
                                    file_record.extracted_text,
                                    source_name=source_name,
                                    should_abort=_check_for_shutdown,
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
                                financial_analysis = analyze_financial_document(
                                    file_record.extracted_text,
                                    source_name=source_name,
                                    should_abort=_check_for_shutdown,
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

                _check_for_shutdown()
                database_item = WorkItem(file_record, correlation_id)
                database_queue.put(database_item)

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
                analysis_duration = time.time() - stage_start

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
                log_unprocessed_file(
                    file_record.file_path,
                    file_record.mime_type,
                    "analysis_failed",
                    str(e),
                )
                with lock:
                    failed_files.add(correlation_id)
            finally:
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
    debug: Annotated[bool, typer.Option(help="Enable debug logging.")] = False,
) -> int:
    """Run the main threaded pipeline."""
    configure_logging(
        level=logging.DEBUG if debug else logging.INFO,
        console=False,
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

    collection = initialize_db(str(DB_PATH))
    run_logger.debug("Database initialized")

    full_manifest = load_manifest()
    run_logger.info("Loaded %d files from manifest", len(full_manifest))

    reset_count = reset_outdated_analysis_tasks(full_manifest)
    if reset_count:
        run_logger.info("Reset %d analysis tasks due to version changes", reset_count)

    # Set up signal handlers for safe shutdown
    global current_manifest
    current_manifest = full_manifest
    run_logger.debug("Manifest reference stored for shutdown handling")

    if target_filename:
        candidate_manifest = [f for f in full_manifest if f.status != COMPLETE]
        run_logger.info(
            "Targeted run enabled; evaluating %d incomplete file(s) for %r.",
            len(candidate_manifest),
            target_filename,
        )
    else:
        candidate_manifest = [f for f in full_manifest if f.status != COMPLETE]
        run_logger.info(
            "Found %d unprocessed files.",
            len(candidate_manifest),
        )

    processing_manifest = candidate_manifest

    if target_filename:
        filtered_manifest = filter_records_by_search_term(
            processing_manifest, target_filename
        )

        if not filtered_manifest:
            fallback_manifest = filter_records_by_search_term(
                full_manifest, target_filename
            )
            if fallback_manifest:
                run_logger.info(
                    "Target %r already complete; reprocessing %d file(s).",
                    target_filename,
                    len(fallback_manifest),
                )
                filtered_manifest = fallback_manifest
            else:
                run_logger.warning(
                    "No files matched the filter %r. Nothing to process.",
                    target_filename,
                )
                return 0

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
    elif batch_size > 0:
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
    for i in range(NUM_EXTRACTION_WORKERS):
        thread = threading.Thread(
            target=extraction_worker, args=(i,), name=f"ExtractionWorker-{i}"
        )
        thread.start()
        threads.append(thread)
        extraction_logger.debug("Started extraction worker %d", i)

    for i in range(NUM_ANALYSIS_WORKERS):
        thread = threading.Thread(
            target=analysis_worker, args=(i, model), name=f"AnalysisWorker-{i}"
        )
        thread.start()
        threads.append(thread)
        analysis_logger.debug("Started analysis worker %d", i)

    for i in range(NUM_DATABASE_WORKERS):
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

    layout = _build_live_layout()
    _update_progress_panel(
        layout,
        completed=len(completed_files),
        failed=len(failed_files),
        remaining=pending_files,
        active=[],
    )

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

    start_time = time.time()
    last_progress_time = start_time
    try:

        with Live(layout, console=LIVE_CONSOLE, refresh_per_second=4):
            with lock:
                completed_count = len(completed_files)
                failed_count = len(failed_files)
                active_files = list(in_progress_files.values())

            initial_remaining = pending_files - (completed_count + failed_count)
            _update_progress_panel(
                layout,
                completed=completed_count,
                failed=failed_count,
                remaining=initial_remaining,
                active=active_files,
            )

            while pending_files > 0 and not shutdown_event.is_set():
                with lock:
                    completed_count = len(completed_files)
                    failed_count = len(failed_files)
                    active_files = list(in_progress_files.values())

                processed_count = completed_count + failed_count
                remaining = pending_files - processed_count

                _update_progress_panel(
                    layout,
                    completed=completed_count,
                    failed=failed_count,
                    remaining=remaining,
                    active=active_files,
                )

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

            _update_progress_panel(
                layout,
                completed=final_completed,
                failed=final_failed,
                remaining=remaining_after,
                active=[],
            )

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

    run_logger.info("Waiting for all work to complete...")
    extraction_queue.join()
    analysis_queue.join()
    database_queue.join()

    run_logger.debug("Sending shutdown signals to workers...")
    dispatch_shutdown_to_workers()

    run_logger.debug("Waiting for worker threads to stop...")
    for thread in threads:
        thread.join(timeout=5)

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
