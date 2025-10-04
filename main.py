#!/usr/bin/env python3
"""Main orchestrator for the file catalog pipeline."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from ai_analyzer import (
    analyze_financial_document,
    analyze_text_content,
    describe_image,
    summarize_video_frames,
)
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


# Constants
MANIFEST_PATH = Path("data/manifest.json")
DB_PATH = Path("data/chromadb")

# Status constants
PENDING_EXTRACTION = "pending_extraction"
PENDING_ANALYSIS = "pending_analysis"
COMPLETE = "complete"


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


def extract_content(file_data: dict) -> None:
    """Extract content from the file based on MIME type."""
    logger = logging.getLogger(__name__)
    mime_type = file_data.get("mime_type", "")
    file_path = file_data["file_path"]

    logger.info("Extracting content from %s (MIME: %s)", file_path, mime_type)

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
        frames = extract_frames_from_video(file_path, "data/frames", interval_sec=10)
        file_data["extracted_frames"] = frames
        # For simplicity, don't extract text from frames here

    # Add more types as needed


def analyze_content(file_data: dict) -> None:
    """Run AI analysis on the extracted content."""
    logger = logging.getLogger(__name__)
    file_path = file_data["file_path"]
    mime_type = file_data.get("mime_type", "")

    logger.info("Analyzing content from %s", file_path)

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

    # Video frame analysis
    elif mime_type.startswith("video/"):
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
                    logger.warning("Failed to analyze frame %s: %s", frame_path, e)
                    continue

            # Combine frame descriptions into video summary
            if frame_descriptions:
                video_summary = summarize_video_frames(frame_descriptions)
                file_data["description"] = video_summary
            else:
                file_data["description"] = "Video frame analysis unavailable"

    # Financial analysis if text looks financial
    if text and ("financial" in text.lower() or "account" in text.lower()):
        financial_analysis = analyze_financial_document(text)
        file_data.update(financial_analysis)
        file_data["has_financial_red_flags"] = bool(
            financial_analysis.get("potential_red_flags")
        )


def main() -> int:
    """Run the main pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("file_catalog.log", mode="a"),
        ],
    )
    # Suppress httpx and related logs
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.info("Starting file catalog pipeline...")

    # Initialize DB
    collection = initialize_db(str(DB_PATH))

    # Load manifest
    manifest = load_manifest()
    logger.info("Loaded %d files from manifest.", len(manifest))

    for file_data in manifest:
        file_path = file_data["file_path"]
        status = file_data.get("status", PENDING_EXTRACTION)

        logger.info("Processing %s (status: %s)", file_path, status)

        if status == PENDING_EXTRACTION:
            extract_content(file_data)
            file_data["status"] = PENDING_ANALYSIS
            status = PENDING_ANALYSIS
            save_manifest(manifest)
            logger.info("Extraction complete for %s", file_path)

        if status == PENDING_ANALYSIS:
            analyze_content(file_data)
            add_file_to_db(file_data, collection)
            file_data["status"] = COMPLETE
            save_manifest(manifest)
            logger.info("Analysis and DB addition complete for %s", file_path)

        elif status == COMPLETE:
            logger.info("Skipping completed file %s", file_path)

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
