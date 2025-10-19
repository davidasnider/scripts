from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import logging
import mimetypes
import shutil
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from src.logging_utils import configure_logging
from src.schema import AnalysisTask, FileRecord
from src.task_utils import determine_analysis_tasks

try:
    import magic  # type: ignore[import-not-found]
except ImportError:
    magic = None  # type: ignore[assignment]

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
    tqdm = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _get_analysis_tasks(mime_type: str, file_path: str = "") -> list[AnalysisTask]:
    return determine_analysis_tasks(mime_type, file_path)


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


def _safe_extract_zip(zip_file: zipfile.ZipFile, target_directory: Path) -> None:
    """Extract ZIP contents while preventing path traversal attacks."""
    target_directory = target_directory.resolve()

    for member in zip_file.infolist():
        member_path = (target_directory / member.filename).resolve()
        if not member_path.is_relative_to(target_directory):
            raise ValueError(f"Unsafe path detected in archive: {member.filename}")

    zip_file.extractall(target_directory)


def _extract_zip_file(zip_path: Path) -> None:
    """Extract a single ZIP archive into its parent directory and remove the archive."""
    logger.info("Extracting %s", zip_path)
    extraction_dir = zip_path.parent

    try:
        with zipfile.ZipFile(zip_path) as archive:
            _safe_extract_zip(archive, extraction_dir)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive {zip_path}: {exc}") from exc

    zip_path.unlink()
    logger.info("Removed extracted archive %s", zip_path)


def _safe_extract_tar(tar_file: tarfile.TarFile, target_directory: Path) -> None:
    """Extract TAR contents while preventing path traversal attacks."""
    target_directory = target_directory.resolve()

    for member in tar_file.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"Disallowed link entry in archive: {member.name}")
        member_path = (target_directory / member.name).resolve()
        if not member_path.is_relative_to(target_directory):
            raise ValueError(f"Unsafe path detected in archive: {member.name}")

    extract_kwargs: dict[str, object] = {}
    if "filter" in inspect.signature(tar_file.extractall).parameters:
        extract_kwargs["filter"] = "data"

    tar_file.extractall(target_directory, **extract_kwargs)


def _extract_tar_archive(tar_path: Path) -> None:
    """Extract a TAR-based archive into its parent directory and remove the archive."""
    logger.info("Extracting %s", tar_path)
    extraction_dir = tar_path.parent

    try:
        with tarfile.open(tar_path, mode="r:*") as archive:
            _safe_extract_tar(archive, extraction_dir)
    except tarfile.TarError as exc:
        raise ValueError(f"Invalid TAR archive {tar_path}: {exc}") from exc

    tar_path.unlink()
    logger.info("Removed extracted archive %s", tar_path)


def _extract_gzip_file(gz_path: Path) -> None:
    """Decompress a single-file GZIP archive and remove the archive."""
    logger.info("Extracting %s", gz_path)
    destination_path = gz_path.with_suffix("")

    if destination_path.exists():
        raise ValueError(
            f"Destination file already exists for gzip archive: {destination_path}"
        )

    with gzip.open(gz_path, "rb") as source, destination_path.open("wb") as target:
        shutil.copyfileobj(source, target)

    gz_path.unlink()
    logger.info("Removed extracted archive %s", gz_path)


def _identify_archive_type(path: Path) -> str | None:
    """Return the archive type string handled by the extractor mapping."""
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".gz"):
        return "gzip"
    return None


ARCHIVE_EXTRACTORS: dict[str, Callable[[Path], None]] = {
    "zip": _extract_zip_file,
    "tar": _extract_tar_archive,
    "gzip": _extract_gzip_file,
}


def _extract_all_archives(root_directory: Path) -> None:
    """Recursively extract supported archives discovered under the given root."""
    extraction_round = 0
    failed_archives: set[Path] = set()

    while True:
        archive_paths: list[tuple[Path, str]] = []
        for path in sorted(root_directory.rglob("*")):
            if not path.is_file() or path in failed_archives:
                continue
            archive_type = _identify_archive_type(path)
            if archive_type is None:
                continue
            archive_paths.append((path, archive_type))

        if not archive_paths:
            if extraction_round == 0:
                logger.info("No archives found under %s", root_directory)
            else:
                logger.info(
                    "Archive extraction complete after %d pass(es)", extraction_round
                )
            if failed_archives:
                logger.warning(
                    "Skipped %d archive(s) due to extraction errors",
                    len(failed_archives),
                )
            break

        extraction_round += 1
        logger.info(
            "Archive extraction pass %d: found %d archive(s)",
            extraction_round,
            len(archive_paths),
        )

        for archive_path, archive_type in archive_paths:
            try:
                extractor = ARCHIVE_EXTRACTORS[archive_type]
                extractor(archive_path)
            except (
                zipfile.BadZipFile,
                tarfile.TarError,
                gzip.BadGzipFile,
                ValueError,
                OSError,
            ) as exc:  # pragma: no cover - depends on filesystem layout
                logger.error("Failed to extract archive %s: %s", archive_path, exc)
                failed_archives.add(archive_path)
            except Exception:  # pragma: no cover - unexpected failure
                logger.exception(
                    "Unexpected error while extracting archive %s", archive_path
                )
                raise


def create_file_manifest(
    root_directory: Path, manifest_path: Path, max_files: int = 0
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

    logger.info("Phase 1: Extracting archives")
    _extract_all_archives(root_directory)

    logger.info("Phase 2: Counting files for progress tracking")
    total_files = _count_files(root_directory)
    if total_files == 0:
        logger.warning("No files found in directory: %s", root_directory)
        return []

    logger.info("Phase 3: Initializing MIME detector")
    mime_detector = _create_mime_detector()
    if mime_detector is not None:
        logger.info("Using python-magic for MIME type detection")
    else:
        logger.info("Using mimetypes fallback for MIME type detection")

    max_limit = max_files if max_files and max_files > 0 else None
    if max_limit is None:
        logger.info("Phase 4: Processing all files (no limit)")
    else:
        logger.info("Phase 4: Processing up to %d files", max_limit)
    start_time = time.time()
    records = []

    try:
        file_iter = _iter_files(root_directory)

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
            # Check if we've reached the total limit
            if max_limit is not None and len(records) >= max_limit:
                logger.info("Reached maximum file limit of %d", max_limit)
                break

            try:
                stat_info = path.stat()
                mime_type = _detect_mime_type(path, mime_detector)

                record = FileRecord(
                    file_path=str(path),
                    file_name=path.name,
                    mime_type=mime_type,
                    file_size=stat_info.st_size,
                    last_modified=stat_info.st_mtime,
                    sha256=_calculate_sha256(path),
                    analysis_tasks=_get_analysis_tasks(mime_type, str(path)),
                )
                records.append(record)
                processed_count += 1

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

    # Log MIME type distribution
    mime_type_counts = Counter(record.mime_type for record in records)
    if mime_type_counts:
        logger.info("File type distribution:")
        for mime_type, count in sorted(mime_type_counts.items()):
            logger.info("  %s: %d files", mime_type, count)

    if error_count > 0:
        logger.warning("Encountered %d errors during processing", error_count)

    logger.info("Phase 5: Writing manifest to %s", manifest_path)
    write_start = time.time()

    manifest_data = [record.model_dump(mode="json") for record in records]

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
        "--max-files",
        type=int,
        default=0,
        help=(
            "Maximum number of files to discover (0 processes all files; default: 0)."
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

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    configure_logging(level=log_level, force=True)

    try:
        manifest = create_file_manifest(
            args.root_directory, args.manifest_path, args.max_files
        )
        logger.info(
            "Successfully wrote %d entries to %s", len(manifest), args.manifest_path
        )

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
