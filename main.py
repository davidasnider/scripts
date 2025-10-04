#!/usr/bin/env python3
"""Main orchestrator for the file catalog pipeline."""

from __future__ import annotations

import dataclasses
import json
import logging
import logging.handlers
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    jsonlogger = None

from ai_analyzer import analyze_financial_document, analyze_text_content, describe_image
from content_extractor import (
    extract_content_from_docx,
    extract_content_from_image,
    extract_content_from_pdf,
    extract_frames_from_video,
)
from database_manager import add_file_to_db, initialize_db
from nsfw_classifier import NSFWClassifier


@dataclasses.dataclass
class DataPacket:
    correlation_id: str
    payload: dict
    metadata: dict = dataclasses.field(default_factory=dict)
    error_info: dict | None = None


def setup_logging() -> logging.handlers.QueueListener:
    """Set up centralized asynchronous logging with JSON formatting."""
    # Create a queue for log messages
    log_queue = queue.Queue()

    # Create a queue handler
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # Create a stream handler for output
    stream_handler = logging.StreamHandler()

    # Set up JSON formatting if available
    if jsonlogger:
        formatter = jsonlogger.JsonFormatter()
    else:
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"message": "%(message)s", "correlation_id": "%(correlation_id)s"}'
        )

    stream_handler.setFormatter(formatter)

    # Create and start the queue listener
    listener = logging.handlers.QueueListener(log_queue, stream_handler)
    listener.start()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)

    return listener


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if yaml is None:
        raise ImportError("PyYAML is required for configuration loading")

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)


# Constants
DB_PATH = Path("data/chromadb")

# Status constants
PENDING_EXTRACTION = "pending_extraction"
PENDING_ANALYSIS = "pending_analysis"
COMPLETE = "complete"


def load_manifest(config: dict) -> list[dict]:
    """Load the manifest from JSON file."""
    manifest_path = Path(config["paths"]["manifest"])
    if not manifest_path.exists():
        return []
    with manifest_path.open() as f:
        return json.load(f)


def save_manifest(manifest: list[dict], config: dict) -> None:
    """Save the manifest to JSON file."""
    manifest_path = Path(config["paths"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)


def extraction_worker(packet: DataPacket) -> DataPacket:
    """Extract content from file data in the packet."""
    log = logging.getLogger(__name__)

    try:
        file_data = packet.payload
        mime_type = file_data.get("mime_type", "")
        file_path = file_data["file_path"]

        log.info(
            f"Extracting content from {file_path} (MIME: {mime_type})",
            extra={"correlation_id": packet.correlation_id},
        )

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

        elif (
            mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            text = extract_content_from_docx(file_path)
            file_data["extracted_text"] = text

        elif mime_type.startswith("image/"):
            text = extract_content_from_image(file_path)
            file_data["extracted_text"] = text

        elif mime_type.startswith("video/"):
            frames = extract_frames_from_video(
                file_path, "data/frames", interval_sec=10
            )
            file_data["extracted_frames"] = frames
            # For simplicity, don't extract text from frames here

        # Update packet status
        file_data["status"] = PENDING_ANALYSIS

        log.info(
            f"Extraction complete for {file_path}",
            extra={"correlation_id": packet.correlation_id},
        )

    except Exception as e:
        log.error(
            f"Extraction failed for {file_path}: {e}",
            extra={"correlation_id": packet.correlation_id},
        )
        packet.error_info = {"stage": "extraction", "error": str(e)}

    return packet


def analysis_worker(packet: DataPacket) -> DataPacket:
    """Run AI analysis on the extracted content in the packet."""
    log = logging.getLogger(__name__)

    try:
        file_data = packet.payload
        file_path = file_data["file_path"]
        mime_type = file_data.get("mime_type", "")

        log.info(
            f"Analyzing content from {file_path}",
            extra={"correlation_id": packet.correlation_id},
        )

        # Text analysis
        text = file_data.get("extracted_text", "")
        if text:
            analysis = analyze_text_content(text)
            file_data.update(analysis)

        # Image description and NSFW
        if mime_type.startswith("image/"):
            description = describe_image(file_path)
            file_data["description"] = description

            classifier = NSFWClassifier()
            nsfw_result = classifier.classify_image(file_path)
            file_data["is_nsfw"] = nsfw_result["label"].lower() == "nsfw"

        # Financial analysis if text looks financial
        if text and ("financial" in text.lower() or "account" in text.lower()):
            financial_analysis = analyze_financial_document(text)
            file_data.update(financial_analysis)
            file_data["has_financial_red_flags"] = bool(
                financial_analysis.get("potential_red_flags")
            )

        # Update packet status
        file_data["status"] = COMPLETE

        log.info(
            f"Analysis complete for {file_path}",
            extra={"correlation_id": packet.correlation_id},
        )

    except Exception as e:
        log.error(
            f"Analysis failed for {file_path}: {e}",
            extra={"correlation_id": packet.correlation_id},
        )
        packet.error_info = {"stage": "analysis", "error": str(e)}

    return packet


def loading_worker(packet: DataPacket, collection) -> DataPacket:
    """Load the processed data into the database."""
    log = logging.getLogger(__name__)

    try:
        file_data = packet.payload
        file_path = file_data["file_path"]

        log.info(
            f"Loading data for {file_path} into database",
            extra={"correlation_id": packet.correlation_id},
        )

        add_file_to_db(file_data, collection)

        log.info(
            f"Database loading complete for {file_path}",
            extra={"correlation_id": packet.correlation_id},
        )

    except Exception as e:
        log.error(
            f"Database loading failed for {file_path}: {e}",
            extra={"correlation_id": packet.correlation_id},
        )
        packet.error_info = {"stage": "loading", "error": str(e)}

    return packet


def extraction_worker_thread(
    extraction_queue: queue.Queue,
    analysis_queue: queue.Queue,
    manifest: list,
    log: logging.Logger,
) -> None:
    """Thread worker for extraction stage."""
    while True:
        try:
            packet = extraction_queue.get(timeout=1)
            if packet is None:  # Sentinel value to stop worker
                break

            file_path = packet.payload.get("file_path", "unknown")
            log.info(
                f"Extracting content from {file_path}",
                extra={"correlation_id": packet.correlation_id},
            )

            packet = extraction_worker(packet)
            if packet.error_info:
                log.error(
                    f"Extraction failed for {file_path}: {packet.error_info}",
                    extra={"correlation_id": packet.correlation_id},
                )
            else:
                # Update status to pending_analysis
                packet.payload["status"] = PENDING_ANALYSIS
                # Update manifest
                for file_data in manifest:
                    if file_data.get("file_path") == file_path:
                        file_data.update(packet.payload)
                        break
                # Send to analysis queue
                analysis_queue.put(packet)

            extraction_queue.task_done()

        except queue.Empty:
            break
        except Exception as e:
            log.error(
                f"Extraction worker error: {e}", extra={"correlation_id": "worker"}
            )


def analysis_worker_thread(
    analysis_queue: queue.Queue,
    loading_queue: queue.Queue,
    manifest: list,
    log: logging.Logger,
) -> None:
    """Thread worker for analysis stage."""
    while True:
        try:
            packet = analysis_queue.get(timeout=1)
            if packet is None:  # Sentinel value to stop worker
                break

            file_path = packet.payload.get("file_path", "unknown")
            log.info(
                f"Analyzing content from {file_path}",
                extra={"correlation_id": packet.correlation_id},
            )

            packet = analysis_worker(packet)
            if packet.error_info:
                log.error(
                    f"Analysis failed for {file_path}: {packet.error_info}",
                    extra={"correlation_id": packet.correlation_id},
                )
            else:
                # Update status to complete
                packet.payload["status"] = COMPLETE
                # Update manifest
                for file_data in manifest:
                    if file_data.get("file_path") == file_path:
                        file_data.update(packet.payload)
                        break
                # Send to loading queue
                loading_queue.put(packet)

            analysis_queue.task_done()

        except queue.Empty:
            break
        except Exception as e:
            log.error(f"Analysis worker error: {e}", extra={"correlation_id": "worker"})


def loading_worker_thread(
    loading_queue: queue.Queue,
    collection,
    manifest: list,
    config: dict,
    log: logging.Logger,
) -> None:
    """Thread worker for loading stage."""
    while True:
        try:
            packet = loading_queue.get(timeout=1)
            if packet is None:  # Sentinel value to stop worker
                break

            file_path = packet.payload.get("file_path", "unknown")
            log.info(
                f"Loading data for {file_path} into database",
                extra={"correlation_id": packet.correlation_id},
            )

            packet = loading_worker(packet, collection)
            if packet.error_info:
                log.error(
                    f"Loading failed for {file_path}: {packet.error_info}",
                    extra={"correlation_id": packet.correlation_id},
                )
            else:
                # Update manifest - need to find the corresponding entry
                for file_data in manifest:
                    if file_data.get("file_path") == file_path:
                        file_data.update(packet.payload)
                        break

            loading_queue.task_done()

        except queue.Empty:
            break
        except Exception as e:
            log.error(f"Loading worker error: {e}", extra={"correlation_id": "worker"})


def main() -> int:
    """Run the main pipeline with concurrent processing."""
    # Load configuration
    config = load_config()

    # Set up logging
    log_listener = setup_logging()
    log = logging.getLogger(__name__)

    log.info("Starting file catalog pipeline", extra={"correlation_id": "system"})

    # Initialize DB
    db_path = config["paths"]["database"]
    collection = initialize_db(db_path)

    # Load manifest
    manifest = load_manifest(config)
    log.info(
        f"Loaded {len(manifest)} files from manifest",
        extra={"correlation_id": "system"},
    )

    # Get worker configuration
    num_workers = config.get("max_workers", {}).get("ingestion", 2)
    analysis_workers = config.get("max_workers", {}).get("analysis", 1)
    loading_workers = config.get("max_workers", {}).get("loading", 1)

    # Create queues for different processing stages
    extraction_queue = queue.Queue()
    analysis_queue = queue.Queue()
    loading_queue = queue.Queue()

    # Populate extraction queue with pending files
    for file_data in manifest:
        status = file_data.get("status", PENDING_EXTRACTION)
        if status in [PENDING_EXTRACTION, PENDING_ANALYSIS]:
            correlation_id = str(uuid.uuid4())
            packet = DataPacket(correlation_id=correlation_id, payload=file_data.copy())
            extraction_queue.put(packet)

    # Add sentinel values to stop workers
    for _ in range(num_workers):
        extraction_queue.put(None)
    for _ in range(analysis_workers):
        analysis_queue.put(None)
    for _ in range(loading_workers):
        loading_queue.put(None)

    # Start concurrent processing
    with ThreadPoolExecutor(
        max_workers=num_workers + analysis_workers + loading_workers
    ) as executor:
        # Submit extraction workers
        extraction_futures = []
        for _ in range(num_workers):
            future = executor.submit(
                extraction_worker_thread,
                extraction_queue,
                analysis_queue,
                manifest,
                log,
            )
            extraction_futures.append(future)

        # Submit analysis workers
        analysis_futures = []
        for _ in range(analysis_workers):
            future = executor.submit(
                analysis_worker_thread, analysis_queue, loading_queue, manifest, log
            )
            analysis_futures.append(future)

        # Submit loading workers
        loading_futures = []
        for _ in range(loading_workers):
            future = executor.submit(
                loading_worker_thread, loading_queue, collection, manifest, config, log
            )
            loading_futures.append(future)

        # Wait for all workers to complete
        for future in as_completed(
            extraction_futures + analysis_futures + loading_futures
        ):
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                log.error(
                    f"Worker thread failed: {e}", extra={"correlation_id": "system"}
                )

    # Save final manifest
    save_manifest(manifest, config)

    log.info("Pipeline complete", extra={"correlation_id": "system"})

    # Stop the log listener
    log_listener.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
