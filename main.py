#!/usr/bin/env python3
"""Main orchestrator for the file catalog pipeline."""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import queue
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

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
from src.nsfw_classifier import NSFWClassifier


class AnalysisModel(str, Enum):
    """Enum for the analysis models."""

    TEXT_ANALYZER = "text_analyzer"
    CODE_ANALYZER = "code_analyzer"
    IMAGE_DESCRIBER = "image_describer"
    ALL = "all"


@dataclasses.dataclass
class WorkItem:
    """Work item for queue-based processing."""

    file_data: dict
    correlation_id: str
    stage: str  # 'extraction', 'analysis', 'database'


@dataclasses.dataclass
class DataPacket:
    """Data packet for passing information between stages."""

    correlation_id: str
    payload: dict
    metadata: dict = dataclasses.field(default_factory=dict)
    error_info: dict | None = None


# Constants
MANIFEST_PATH = Path("data/manifest.json")
DB_PATH = Path("data/chromadb")

# Status constants
PENDING_EXTRACTION = "pending_extraction"
PENDING_ANALYSIS = "pending_analysis"
COMPLETE = "complete"

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


def load_manifest() -> list[dict]:
    """Load the manifest from JSON file."""
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open() as f:
        return json.load(f)


def save_manifest(manifest: list[dict]) -> None:
    """Save the manifest to JSON file."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)


def extraction_worker(worker_id: int) -> None:
    """Worker thread for content extraction."""
    logger = logging.getLogger("main")
    logger.debug("Extraction worker %d started", worker_id)

    while True:
        work_item = None
        try:
            work_item = extraction_queue.get(timeout=1.0)
            if work_item is None:  # Poison pill
                logger.debug("Extraction worker %d received shutdown signal", worker_id)
                extraction_queue.task_done()  # Mark the poison pill as done
                break

            file_data = work_item.file_data
            correlation_id = work_item.correlation_id
            mime_type = file_data.get("mime_type", "")
            file_path = file_data["file_path"]

            logger.debug(
                "Worker %d extracting content from %s (MIME: %s) [%s]",
                worker_id,
                file_path,
                mime_type,
                correlation_id,
            )

            try:
                if mime_type.startswith("text/") or mime_type == "application/pdf":
                    if mime_type == "application/pdf":
                        text = extract_content_from_pdf(file_path)
                    else:
                        # For plain text, just read it
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                text = f.read()
                        except Exception:
                            text = ""
                    file_data["extracted_text"] = text

                elif mime_type == (
                    "application/vnd.openxmlformats-"
                    "officedocument.wordprocessingml.document"
                ):
                    text = extract_content_from_docx(file_path)
                    file_data["extracted_text"] = text

                elif mime_type == (
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ):
                    text = extract_content_from_xlsx(file_path)
                    file_data["extracted_text"] = text

                elif mime_type.startswith("image/"):
                    text = extract_content_from_image(file_path)
                    file_data["extracted_text"] = text

                elif mime_type.startswith("video/"):
                    frames = extract_frames_from_video(
                        file_path, "data/frames", interval_sec=10
                    )
                    file_data["extracted_frames"] = frames

                # Update status and move to analysis queue
                file_data["status"] = PENDING_ANALYSIS
                analysis_item = WorkItem(file_data, correlation_id, "analysis")
                analysis_queue.put(analysis_item)

                logger.debug(
                    "Worker %d completed extraction for %s [%s]",
                    worker_id,
                    file_path,
                    correlation_id,
                )

            except Exception as e:
                logger.error(
                    "Worker %d failed to extract content from %s [%s]: %s",
                    worker_id,
                    file_path,
                    correlation_id,
                    e,
                )
                with lock:
                    failed_files.add(correlation_id)

            extraction_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error("Extraction worker %d error: %s", worker_id, e)
            # Don't call task_done() here as it's already handled above

    logger.debug("Extraction worker %d stopped", worker_id)


def analysis_worker(worker_id: int, model: AnalysisModel) -> None:
    """Worker thread for AI analysis."""
    logger = logging.getLogger(__name__)
    logger.debug("Analysis worker %d started", worker_id)

    while True:
        work_item = None
        try:
            work_item = analysis_queue.get(timeout=1.0)
            if work_item is None:  # Poison pill
                logger.debug("Analysis worker %d received shutdown signal", worker_id)
                analysis_queue.task_done()  # Mark the poison pill as done
                break

            file_data = work_item.file_data
            correlation_id = work_item.correlation_id
            file_path = file_data["file_path"]
            mime_type = file_data.get("mime_type", "")

            logger.debug(
                "Worker %d analyzing content from %s [%s]",
                worker_id,
                file_path,
                correlation_id,
            )

            # Process the work item
            try:
                # Text analysis
                if model in (AnalysisModel.ALL, AnalysisModel.TEXT_ANALYZER):
                    text = file_data.get("extracted_text", "")
                    if text:
                        logger.debug(
                            "Worker %d running text analysis for %s [%s]",
                            worker_id,
                            file_path,
                            correlation_id,
                        )
                        analysis = analyze_text_content(text)
                        file_data.update(analysis)

                # Image description and NSFW
                if model in (AnalysisModel.ALL, AnalysisModel.IMAGE_DESCRIBER):
                    if mime_type.startswith("image/"):
                        logger.debug(
                            "Worker %d running image analysis for %s [%s]",
                            worker_id,
                            file_path,
                            correlation_id,
                        )
                        description = describe_image(file_path)
                        file_data["description"] = description

                        # Also analyze image description for summary & mentioned_people
                        logger.debug(
                            "Worker %d analyzing image description for %s [%s]",
                            worker_id,
                            file_path,
                            correlation_id,
                        )
                        image_text_analysis = analyze_text_content(description)
                        file_data.update(image_text_analysis)

                        classifier = NSFWClassifier()
                        nsfw_result = classifier.classify_image(file_path)
                        file_data["is_nsfw"] = nsfw_result["label"].lower() == "nsfw"

                # Video frame analysis
                if model in (AnalysisModel.ALL, AnalysisModel.IMAGE_DESCRIBER):
                    if mime_type.startswith("video/"):
                        logger.debug(
                            "Worker %d running video analysis for %s [%s]",
                            worker_id,
                            file_path,
                            correlation_id,
                        )
                        frames = file_data.get("extracted_frames", [])
                        if frames:
                            frame_descriptions = []
                            classifier = NSFWClassifier()
                            for frame_path in frames:
                                try:
                                    description = describe_image(frame_path)
                                    frame_descriptions.append(description)

                                    # Check NSFW for each frame
                                    nsfw_result = classifier.classify_image(frame_path)
                                    if nsfw_result["label"].lower() == "nsfw":
                                        file_data["is_nsfw"] = True
                                except Exception as e:
                                    logger.warning(
                                        "Worker %d failed to analyze frame %s [%s]: %s",
                                        worker_id,
                                        frame_path,
                                        correlation_id,
                                        e,
                                    )
                                    continue

                            # Combine frame descriptions into video summary
                            if frame_descriptions:
                                video_summary = summarize_video_frames(
                                    frame_descriptions
                                )
                                file_data["description"] = video_summary

                                # Analyze video summary for summary & mentioned_people
                                logger.debug(
                                    "Worker %d analyzing video summary for %s [%s]",
                                    worker_id,
                                    file_path,
                                    correlation_id,
                                )
                                video_text_analysis = analyze_text_content(
                                    video_summary
                                )
                                file_data.update(video_text_analysis)
                            else:
                                file_data["description"] = (
                                    "Video frame analysis unavailable"
                                )
                                file_data["summary"] = (
                                    "Video frame analysis unavailable"
                                )
                                file_data["mentioned_people"] = []

                # Financial analysis if text looks financial
                if model in (AnalysisModel.ALL, AnalysisModel.CODE_ANALYZER):
                    text = file_data.get("extracted_text", "")
                    if text and (
                        "financial" in text.lower() or "account" in text.lower()
                    ):
                        logger.debug(
                            "Worker %d running financial analysis for %s [%s]",
                            worker_id,
                            file_path,
                            correlation_id,
                        )
                        financial_analysis = analyze_financial_document(text)
                        file_data.update(financial_analysis)
                        file_data["has_financial_red_flags"] = bool(
                            financial_analysis.get("potential_red_flags")
                        )

                # Ensure all files have consistent summary and description fields
                if "summary" not in file_data:
                    file_data["summary"] = file_data.get(
                        "description", "No summary available"
                    )
                if "description" not in file_data:
                    file_data["description"] = file_data.get(
                        "summary", "No description available"
                    )
                if "mentioned_people" not in file_data:
                    file_data["mentioned_people"] = []

                # Move to database queue
                database_item = WorkItem(file_data, correlation_id, "database")
                database_queue.put(database_item)

                logger.debug(
                    "Worker %d completed analysis for %s [%s]",
                    worker_id,
                    file_path,
                    correlation_id,
                )

            except Exception as e:
                logger.error(
                    "Worker %d failed to analyze content from %s [%s]: %s",
                    worker_id,
                    file_path,
                    correlation_id,
                    e,
                )
                with lock:
                    failed_files.add(correlation_id)
                # Failure recorded in failed_files set above

            # Always mark the work item as done
            analysis_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error("Analysis worker %d error: %s", worker_id, e)
            # Don't call task_done() here as it's already handled above

    logger.debug("Analysis worker %d stopped", worker_id)


def database_worker(worker_id: int, collection: Any) -> None:
    """Worker thread for database operations."""
    logger = logging.getLogger(__name__)
    logger.debug("Database worker %d started", worker_id)

    while True:
        work_item = None
        try:
            work_item = database_queue.get(timeout=1.0)
            if work_item is None:  # Poison pill
                logger.debug("Database worker %d received shutdown signal", worker_id)
                database_queue.task_done()  # Mark the poison pill as done
                break

            file_data = work_item.file_data
            correlation_id = work_item.correlation_id
            file_path = file_data["file_path"]

            logger.debug(
                "Worker %d adding to database: %s [%s]",
                worker_id,
                file_path,
                correlation_id,
            )

            try:
                add_file_to_db(file_data, collection)
                file_data["status"] = COMPLETE

                with lock:
                    completed_files.add(correlation_id)

                logger.debug(
                    "Worker %d completed database addition for %s [%s]",
                    worker_id,
                    file_path,
                    correlation_id,
                )

            except Exception as e:
                logger.error(
                    "Worker %d failed to add %s to database [%s]: %s",
                    worker_id,
                    file_path,
                    correlation_id,
                    e,
                )
                with lock:
                    failed_files.add(correlation_id)

            database_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error("Database worker %d error: %s", worker_id, e)
            # Don't call task_done() here as it's already handled above

    logger.debug("Database worker %d stopped", worker_id)


def save_manifest_periodically(manifest: list[dict]) -> None:
    """Periodically save manifest to persist progress."""
    logger = logging.getLogger(__name__)
    while True:
        time.sleep(10)  # Save every 10 seconds
        try:
            save_manifest(manifest)
            logger.debug("Manifest saved periodically")
        except Exception as e:
            logger.warning("Failed to save manifest: %s", e)


def write_processed_files_to_csv(processed_files: list[dict], file_path: str) -> None:
    """Write processed file data to a CSV file."""
    if not processed_files:
        return

    # Define the headers based on the keys of the first dictionary
    headers = processed_files[0].keys()

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(processed_files)


def main(
    model: Annotated[
        AnalysisModel, typer.Option(help="The model to use for analysis.")
    ] = AnalysisModel.ALL,
    limit: Annotated[
        int, typer.Option(help="Limit the number of files to process.")
    ] = 0,
    csv_output: Annotated[
        str,
        typer.Option(
            help="Path to save a CSV file of processed files.", rich_help_panel="Output"
        ),
    ] = "",
) -> int:
    """Run the main threaded pipeline."""
    # Set up logging with INFO level
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("file_catalog.log", mode="a"),
        ],
    )
    # Suppress httpx and related logs but keep some visibility
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Suppress transformers/huggingface warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    # Suppress specific warnings we don't need to see
    import warnings

    warnings.filterwarnings("ignore", message=".*use_fast.*")
    warnings.filterwarnings("ignore", message=".*slow processor.*")
    warnings.filterwarnings("ignore", message=".*Device set to use.*")

    # Suppress MuPDF error messages about Screen annotations
    import sys

    # Redirect stderr to suppress specific noisy MuPDF errors
    MUPDF_SCREEN_ERR_SUBSTR = (
        "MuPDF error: unsupported error: cannot create appearance stream for Screen"
    )

    class MuPDFErrorFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr

        def write(self, data):
            # Filter out MuPDF Screen annotation appearance errors
            if MUPDF_SCREEN_ERR_SUBSTR not in data:
                self.original_stderr.write(data)

        def flush(self):
            self.original_stderr.flush()

    # Apply the filter
    sys.stderr = MuPDFErrorFilter(sys.stderr)
    logger = logging.getLogger(__name__)
    logger.info("Starting threaded file catalog pipeline...")

    # Initialize DB
    collection = initialize_db(str(DB_PATH))
    logger.debug("Database initialized")

    # Load manifest
    manifest = load_manifest()
    logger.info("Loaded %d files from manifest", len(manifest))

    if limit > 0:
        manifest = manifest[:limit]
        logger.info("Limiting to %d files", len(manifest))

    # Start periodic manifest saver
    manifest_saver = threading.Thread(
        target=save_manifest_periodically, args=(manifest,), daemon=True
    )
    manifest_saver.start()
    logger.debug("Started manifest saver thread")

    # Start worker threads
    threads = []

    # Extraction workers
    for i in range(NUM_EXTRACTION_WORKERS):
        thread = threading.Thread(
            target=extraction_worker, args=(i,), name=f"ExtractionWorker-{i}"
        )
        thread.start()
        threads.append(thread)
        logger.debug("Started extraction worker %d", i)

    # Analysis workers
    for i in range(NUM_ANALYSIS_WORKERS):
        thread = threading.Thread(
            target=analysis_worker, args=(i, model), name=f"AnalysisWorker-{i}"
        )
        thread.start()
        threads.append(thread)
        logger.debug("Started analysis worker %d", i)

    # Database workers
    for i in range(NUM_DATABASE_WORKERS):
        thread = threading.Thread(
            target=database_worker, args=(i, collection), name=f"DatabaseWorker-{i}"
        )
        thread.start()
        threads.append(thread)
        logger.debug("Started database worker %d", i)

    logger.info("All worker threads started")

    # Queue work items based on current status
    total_files = len(manifest)
    pending_files = 0

    for file_data in manifest:
        file_path = file_data["file_path"]
        status = file_data.get("status", PENDING_EXTRACTION)
        correlation_id = f"file-{hash(file_path) % 100000:05d}"

        logger.debug("Queuing %s (status: %s) [%s]", file_path, status, correlation_id)

        if status == PENDING_EXTRACTION:
            work_item = WorkItem(file_data, correlation_id, "extraction")
            extraction_queue.put(work_item)
            pending_files += 1

        elif status == PENDING_ANALYSIS:
            work_item = WorkItem(file_data, correlation_id, "analysis")
            analysis_queue.put(work_item)
            pending_files += 1

        elif status == COMPLETE:
            logger.debug("Skipping completed file %s [%s]", file_path, correlation_id)
            with lock:
                completed_files.add(correlation_id)

    logger.info("Queued %d pending files for processing", pending_files)

    # Monitor progress
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
        if current_time - last_progress_time >= 30:  # Log progress every 30 seconds
            elapsed = current_time - start_time
            if processed_count > 0:
                estimated_total = elapsed * pending_files / processed_count
                estimated_remaining = estimated_total - elapsed
                logger.info(
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
                logger.info(
                    "Progress: %d/%d completed, %d failed, %d remaining",
                    completed_count,
                    pending_files,
                    failed_count,
                    remaining,
                )
            last_progress_time = current_time

        if remaining <= 0:
            break

    # Wait for all queues to finish
    logger.info("Waiting for all work to complete...")
    extraction_queue.join()
    analysis_queue.join()
    database_queue.join()

    # Send poison pills to stop workers
    logger.debug("Sending shutdown signals to workers...")
    for _ in range(NUM_EXTRACTION_WORKERS):
        extraction_queue.put(None)
    for _ in range(NUM_ANALYSIS_WORKERS):
        analysis_queue.put(None)
    for _ in range(NUM_DATABASE_WORKERS):
        database_queue.put(None)

    # Wait for all worker threads to finish
    logger.debug("Waiting for worker threads to stop...")
    for thread in threads:
        thread.join()

    # Final save
    save_manifest(manifest)

    # Final statistics
    elapsed_time = time.time() - start_time
    with lock:
        completed_count = len(completed_files)
        failed_count = len(failed_files)

    logger.info(
        "Pipeline complete! Processed %d files in %.1f seconds "
        "(%d completed, %d failed)",
        total_files,
        elapsed_time,
        completed_count,
        failed_count,
    )

    # Write to CSV if a path is provided
    if csv_output:
        processed_files_data = [
            item for item in manifest if item.get("status") == COMPLETE
        ]
        if processed_files_data:
            logger.info(
                "Writing %d processed files to %s",
                len(processed_files_data),
                csv_output,
            )
            write_processed_files_to_csv(processed_files_data, csv_output)
        else:
            logger.info("No files to write to CSV.")

    if failed_count > 0:
        logger.warning("Some files failed to process. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    typer.run(main)
