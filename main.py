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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
import typer
from typing_extensions import Annotated

from src.ai_analyzer import (
    analyze_financial_document,
    analyze_text_content,
    describe_image,
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
from src.manifest_utils import reset_outdated_analysis_tasks
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
}


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
lock = threading.Lock()

# Global manifest for signal handlers
current_manifest = None

LOGGER_NAME = "file_catalog.pipeline"
pipeline_logger = logging.getLogger(LOGGER_NAME)
extraction_logger = logging.getLogger(f"{LOGGER_NAME}.extraction")
analysis_logger = logging.getLogger(f"{LOGGER_NAME}.analysis")
database_logger = logging.getLogger(f"{LOGGER_NAME}.database")


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM by saving manifest before exit."""
    pipeline_logger.info("Received signal %d, saving manifest before exit...", signum)

    if current_manifest is not None:
        try:
            save_manifest(current_manifest)
            pipeline_logger.info("Manifest saved successfully")
        except Exception as e:
            pipeline_logger.error("Failed to save manifest on exit: %s", e)

    # Exit gracefully
    sys.exit(1)


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
                stage_start = time.time()
                extracted_frames: list[str] = []
                extracted_text = ""

                if (
                    file_record.mime_type.startswith("text/")
                    or file_record.mime_type == "application/pdf"
                ):
                    if file_record.mime_type == "application/pdf":
                        extracted_text = extract_content_from_pdf(file_record.file_path)
                    else:
                        try:
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
                    extracted_text = extract_content_from_docx(file_record.file_path)
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type == (
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ):
                    extracted_text = extract_content_from_xlsx(file_record.file_path)
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type == "image/svg+xml":
                    extracted_text = extract_text_from_svg(file_record.file_path)
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type.startswith("image/"):
                    extracted_text = extract_content_from_image(file_record.file_path)
                    file_record.extracted_text = extracted_text

                elif file_record.mime_type.startswith(
                    "video/"
                ) and not file_record.file_path.lower().endswith(".asx"):
                    extracted_frames = extract_frames_from_video(
                        file_record.file_path, "data/frames", interval_sec=10
                    )
                    file_record.extracted_frames = extracted_frames

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
                analysis_item = WorkItem(file_record, correlation_id)
                analysis_queue.put(analysis_item)

                worker_logger.debug(
                    "Worker %d completed extraction for %s [%s]",
                    worker_id,
                    file_record.file_path,
                    correlation_id,
                )

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

            worker_logger.debug(
                "Worker %d analyzing content from %s [%s]",
                worker_id,
                file_record.file_path,
                correlation_id,
            )

            try:
                stage_start = time.time()

                text_analysis_result: dict[str, Any] | None = None

                for task in file_record.analysis_tasks:
                    if task.status == AnalysisStatus.PENDING:
                        if (
                            model != AnalysisModel.ALL
                            and task.name.value != model.value
                        ):
                            continue

                        try:
                            if task.name in TEXT_BASED_ANALYSES:
                                if file_record.extracted_text:
                                    if text_analysis_result is None:
                                        text_analysis_result = analyze_text_content(
                                            file_record.extracted_text
                                        )
                                    if task.name == AnalysisName.TEXT_ANALYSIS:
                                        file_record.summary = text_analysis_result.get(
                                            "summary"
                                        )
                                    else:
                                        file_record.mentioned_people = (
                                            text_analysis_result.get(
                                                "mentioned_people", []
                                            )
                                        )

                            elif task.name == AnalysisName.IMAGE_DESCRIPTION:
                                if file_record.mime_type.startswith("image/"):
                                    description = describe_image(file_record.file_path)
                                    file_record.description = description
                                    if not file_record.summary:
                                        file_record.summary = description

                            elif task.name == AnalysisName.NSFW_CLASSIFICATION:
                                if file_record.mime_type.startswith("image/"):
                                    classifier = NSFWClassifier()
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
                                            nsfw_result = classifier.classify_image(
                                                frame_path
                                            )
                                            if nsfw_result["label"].lower() == "nsfw":
                                                file_record.is_nsfw = True
                                                break

                            elif task.name == AnalysisName.VIDEO_SUMMARY:
                                if file_record.mime_type.startswith(
                                    "video/"
                                ) and not file_record.file_path.lower().endswith(
                                    ".asx"
                                ):
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
                                            "Video too short or frame "
                                            "extraction failed",
                                        )
                                        # Still mark as complete since there's
                                        # nothing to summarize
                                    else:
                                        frame_descriptions = []
                                        for frame_path in file_record.extracted_frames:
                                            try:
                                                description = describe_image(frame_path)
                                                frame_descriptions.append(description)
                                            except Exception as e:
                                                worker_logger.warning(
                                                    "Failed to describe frame %s: %s",
                                                    frame_path,
                                                    e,
                                                )
                                        if frame_descriptions:
                                            video_summary = summarize_video_frames(
                                                frame_descriptions
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
                                    financial_analysis = analyze_financial_document(
                                        file_record.extracted_text
                                    )
                                    file_record.summary = financial_analysis.get(
                                        "summary"
                                    )
                                    file_record.potential_red_flags = (
                                        financial_analysis.get("potential_red_flags")
                                    )
                                    file_record.incriminating_items = (
                                        financial_analysis.get("incriminating_items")
                                    )
                                    file_record.confidence_score = (
                                        financial_analysis.get("confidence_score")
                                    )
                                    file_record.has_financial_red_flags = bool(
                                        financial_analysis.get("potential_red_flags")
                                    )

                            task.status = AnalysisStatus.COMPLETE

                        except Exception as e:
                            worker_logger.error(
                                "Failed to run analysis task %s for %s: %s",
                                task.name,
                                file_record.file_path,
                                e,
                            )
                            task.status = AnalysisStatus.ERROR
                            task.error_message = str(e)

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

            analysis_queue.task_done()

        except queue.Empty:
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
        force=True,
    )

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
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    run_logger.debug("Signal handlers installed for safe manifest saving")

    if batch_size > 0:
        unprocessed_files = [f for f in full_manifest if f.status != COMPLETE]
        processing_manifest = unprocessed_files[:batch_size]
        run_logger.info(
            "Found %d unprocessed files. Processing batch of %d files.",
            len(unprocessed_files),
            len(processing_manifest),
        )
    else:
        processing_manifest = [f for f in full_manifest if f.status != COMPLETE]
        run_logger.info(
            "Found %d unprocessed files. Processing all remaining files.",
            len(processing_manifest),
        )

    if target_filename:
        search_term = target_filename.lower()
        filtered_manifest = [
            record
            for record in processing_manifest
            if search_term in record.file_path.lower()
            or Path(record.file_path).name.lower() == search_term
        ]

        if not filtered_manifest:
            run_logger.warning(
                "No files matched the filter %r. Nothing to process.",
                target_filename,
            )
            return 0

        processing_manifest = filtered_manifest
        run_logger.info(
            "Filtered manifest to %d file(s) matching %r.",
            len(processing_manifest),
            target_filename,
        )

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
        elif (
            model is AnalysisModel.TEXT_ANALYZER
            or model is AnalysisModel.PEOPLE_ANALYZER
        ):
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

    try:
        start_time = time.time()
        last_progress_time = start_time

        while pending_files > 0:
            time.sleep(2)

            with lock:
                completed_count = len(completed_files)
                failed_count = len(failed_files)

            processed_count = completed_count + failed_count
            remaining = pending_files - processed_count

            current_time = time.time()
            if current_time - last_progress_time >= 30:
                elapsed = current_time - start_time
                if processed_count > 0:
                    estimated_total = elapsed * pending_files / processed_count
                    estimated_remaining = estimated_total - elapsed
                    run_logger.info(
                        "Progress: %d/%d completed, %d failed, %d remaining "
                        "(%.1f%% complete, ~%.1f minutes remaining)",
                        completed_count,
                        pending_files,
                        failed_count,
                        remaining,
                        (processed_count / pending_files) * 100,
                        estimated_remaining / 60,
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

        run_logger.info("Waiting for all work to complete...")
        extraction_queue.join()
        analysis_queue.join()
        database_queue.join()

    finally:
        # Ensure manifest is saved even if there's an unexpected exception
        run_logger.info("Ensuring manifest is saved...")
        try:
            save_manifest(full_manifest)
            run_logger.info("Manifest saved successfully")
        except Exception as e:
            run_logger.error("Failed to save manifest: %s", e)

    run_logger.debug("Sending shutdown signals to workers...")
    for _ in range(NUM_EXTRACTION_WORKERS):
        extraction_queue.put(None)
    for _ in range(NUM_ANALYSIS_WORKERS):
        analysis_queue.put(None)
    for _ in range(NUM_DATABASE_WORKERS):
        database_queue.put(None)

    run_logger.debug("Waiting for worker threads to stop...")
    for thread in threads:
        thread.join(timeout=5)

    save_manifest(full_manifest)

    elapsed_time = time.time() - start_time
    with lock:
        completed_count = len(completed_files)
        failed_count = len(failed_files)

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
                        f"incomplete_{task.name.value}",
                        f"Task failed or skipped: "
                        f"{task.error_message or 'Unknown reason'}",
                    )

    if incomplete_files:
        run_logger.info(
            "Found %d files with incomplete analysis tasks. "
            "Check data/unprocessed_files.csv for details.",
            len(incomplete_files),
        )

    if csv_output:
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

    if failed_count > 0:
        run_logger.warning("Some files failed to process. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    typer.run(main)
