from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import magic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled gracefully at runtime
    magic = None  # type: ignore[assignment]

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled gracefully at runtime
    tqdm = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


STATUS_PENDING_EXTRACTION = "pending_extraction"


@dataclass(frozen=True)
class FileRecord:
    file_path: str
    file_name: str
    mime_type: str
    size_bytes: int
    status: str = STATUS_PENDING_EXTRACTION

    @classmethod
    def from_path(cls, path: Path, mime_detector: Any) -> "FileRecord":
        mime_type = _detect_mime_type(path, mime_detector)

        stat_info = path.stat()
        return cls(
            file_path=str(path),
            file_name=path.name,
            mime_type=mime_type,
            size_bytes=stat_info.st_size,
        )


def _count_files(root_directory: Path) -> int:
    """Count total files for progress tracking."""
    count = 0
    logger.info("Counting files in %s...", root_directory)
    start_time = time.time()

    for path in root_directory.rglob("*"):
        if path.is_file():
            count += 1

    elapsed = time.time() - start_time
    logger.info("Found %d files in %.2f seconds", count, elapsed)
    return count


def _iter_files(root_directory: Path) -> Iterable[Path]:
    for path in sorted(root_directory.rglob("*")):
        if path.is_file():
            yield path


def _create_mime_detector() -> Any:
    if magic is None:
        return None

    try:
        return magic.Magic(mime=True)
    except Exception as exc:  # pragma: no cover - depends on system libmagic
        logger.warning(
            "python-magic could not initialize libmagic (%s). "
            "Falling back to mimetypes-based detection.",
            exc,
        )
        return None


def _detect_mime_type(path: Path, mime_detector: Any) -> str:
    if mime_detector is not None:
        try:
            mime_type = mime_detector.from_file(str(path))
            logger.debug("Detected MIME type for %s: %s", path.name, mime_type)
            return mime_type
        except Exception as exc:
            logger.debug("Magic detection failed for %s: %s", path.name, exc)

    guessed_type, _ = mimetypes.guess_type(str(path))
    if guessed_type:
        logger.debug("Guessed MIME type for %s: %s", path.name, guessed_type)
        return guessed_type

    logger.debug("Using fallback MIME type for %s", path.name)
    return "application/octet-stream"


def create_file_manifest(
    root_directory: Path, manifest_path: Path
) -> list[dict[str, object]]:
    logger.info("Starting file discovery process")
    logger.info("Root directory: %s", root_directory)
    logger.info("Manifest output: %s", manifest_path)

    root_directory = root_directory.expanduser().resolve()
    if not root_directory.is_dir():
        raise ValueError(
            f"Root directory does not exist or is not a directory: {root_directory}"
        )

    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Created manifest directory: %s", manifest_path.parent)

    # Phase 1: Count files for progress tracking
    logger.info("Phase 1: Counting files for progress tracking")
    total_files = _count_files(root_directory)
    if total_files == 0:
        logger.warning("No files found in directory: %s", root_directory)
        return []

    # Phase 2: Initialize MIME detector
    logger.info("Phase 2: Initializing MIME detector")
    mime_detector = _create_mime_detector()
    if mime_detector is not None:
        logger.info("Using python-magic for MIME type detection")
    else:
        logger.info("Using mimetypes fallback for MIME type detection")

    # Phase 3: Process files with progress tracking
    logger.info("Phase 3: Processing %d files", total_files)
    start_time = time.time()
    records = []

    try:
        file_iter = _iter_files(root_directory)

        # Use tqdm if available, otherwise process without progress bar
        if tqdm is not None:
            file_iter = tqdm(
                file_iter,
                total=total_files,
                desc="Discovering files",
                unit="files",
                dynamic_ncols=True,
            )

        processed_count = 0
        error_count = 0

        for path in file_iter:
            try:
                record = FileRecord.from_path(path, mime_detector)
                records.append(record)
                processed_count += 1

                # Log progress every 100 files if no tqdm
                if tqdm is None and processed_count % 100 == 0:
                    logger.info("Processed %d/%d files", processed_count, total_files)

            except Exception as exc:
                error_count += 1
                logger.error("Failed to process file %s: %s", path, exc)
                continue

    finally:
        if mime_detector is not None and hasattr(mime_detector, "close"):
            try:
                mime_detector.close()
                logger.debug("Closed MIME detector")
            except Exception as exc:
                logger.warning("Failed to close MIME detector: %s", exc)

    processing_time = time.time() - start_time
    logger.info("Processed %d files in %.2f seconds", len(records), processing_time)

    if error_count > 0:
        logger.warning("Encountered %d errors during processing", error_count)

    # Phase 4: Write manifest
    logger.info("Phase 4: Writing manifest to %s", manifest_path)
    write_start = time.time()

    manifest_data = [record.__dict__ for record in records]

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest_data, manifest_file, indent=2)
        manifest_file.write("\n")

    write_time = time.time() - write_start
    total_time = time.time() - start_time

    logger.info("Manifest written in %.2f seconds", write_time)
    logger.info("Total processing time: %.2f seconds", total_time)
    logger.info("Successfully created manifest with %d entries", len(manifest_data))

    return manifest_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discover-files",
        description=(
            "Create a manifest describing files rooted at the specified directory."
        ),
    )
    parser.add_argument(
        "root_directory",
        type=Path,
        help="Directory to recursively scan for files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifest.json"),
        help=(
            "Location to write the generated manifest JSON "
            "(default: data/manifest.json)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging output."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging output."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on verbosity level
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        manifest = create_file_manifest(args.root_directory, args.manifest_path)
        logger.info(
            "Successfully wrote %d entries to %s", len(manifest), args.manifest_path
        )

        # Print summary to stdout for non-verbose runs
        if not args.verbose and not args.debug:
            print(
                f"Discovered {len(manifest)} files and wrote manifest to "
                f"{args.manifest_path}"
            )

        return 0
    except Exception as exc:
        logger.error("Failed to create file manifest: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
