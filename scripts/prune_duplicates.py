#!/usr/bin/env python3
"""Remove duplicate files listed in the manifest by pruning matching SHA entries."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

try:
    from src.logging_utils import configure_logging  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script runs
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_ROOT = PROJECT_ROOT / "src"
    for candidate in (PROJECT_ROOT, SRC_ROOT):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.append(candidate_str)
    from src.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger("file_catalog.prune_duplicates")

ManifestEntry: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class DuplicateEntry:
    """Represents a manifest entry to delete and the canonical copy to retain."""

    entry_index: int
    entry: ManifestEntry
    keep_index: int
    keep_entry: ManifestEntry


def sort_key_for_entry(entry: ManifestEntry, *, sha: str) -> tuple[int, str]:
    """Generate a stable sort key for duplicate selection."""
    path_str = entry.get("file_path")
    if not isinstance(path_str, str):
        logger.warning(
            "SHA %s: skipping duplicate entry lacking string file_path: %s",
            sha,
            entry,
        )
        return (sys.maxsize, "")
    return (len(path_str), path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove duplicate files referenced in the manifest when their "
            "SHA-256 hashes match."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.json"),
        help="Path to the manifest JSON file (default: data/manifest.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview duplicate pruning actions without deleting files or modifying "
            "the manifest."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level to use (default: INFO).",
    )
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Create a backup of the original manifest before writing changes "
            "(default: enabled)."
        ),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[ManifestEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}")
    with path.open("r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)
    if not isinstance(data, list):
        raise ValueError(f"Manifest at {path} must be a JSON array.")
    return data


def group_duplicates(
    entries: list[ManifestEntry],
) -> dict[str, list[tuple[int, ManifestEntry]]]:
    """Return SHA buckets of manifest entries, with each value keeping its index.

    Non-string or missing `sha256` values are ignored. Only groups containing more
    than one entry are returned so callers can focus on duplicate candidates.
    """
    groups: dict[str, list[tuple[int, ManifestEntry]]] = {}
    for idx, entry in enumerate(entries):
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not sha:
            logger.debug("Skipping entry without sha256: %s", entry)
            continue
        groups.setdefault(sha, []).append((idx, entry))
    return {sha: items for sha, items in groups.items() if len(items) > 1}


def select_removals(
    duplicate_groups: dict[str, list[tuple[int, ManifestEntry]]],
) -> list[DuplicateEntry]:
    """Pick manifest entries to remove for each duplicate SHA group.

    Preference order:
    1. Entries whose `file_path` exists on disk.
    2. Within that subset (or all candidates if none exist), the shortest path length
       followed by lexicographic order to keep the most canonical-looking location.
    Each removal entry records both the duplicate manifest index and the retained
    entry so downstream logic can update the manifest and optionally delete files.
    """
    removals: list[DuplicateEntry] = []
    for sha, items in duplicate_groups.items():
        candidates = [
            (idx, entry)
            for idx, entry in items
            if isinstance(entry.get("file_path"), str)
        ]
        if not candidates:
            logger.warning(
                "SHA %s has duplicates but none include a valid file_path; skipping.",
                sha,
            )
            continue
        existing_candidates = [
            (idx, entry)
            for idx, entry in candidates
            if Path(entry["file_path"]).exists()
        ]

        preferred_pool = existing_candidates or candidates
        if existing_candidates and len(existing_candidates) != len(candidates):
            logger.info(
                "SHA %s: preferring among %d existing path(s) out of %d candidates.",
                sha,
                len(existing_candidates),
                len(candidates),
            )

        keep_index, keep_entry_dict = min(
            preferred_pool,
            key=lambda pair: sort_key_for_entry(pair[1], sha=sha),
        )
        keep_path = keep_entry_dict.get("file_path")
        if not isinstance(keep_path, str):
            logger.warning(
                "SHA %s: could not identify a valid file_path to retain; skipping.",
                sha,
            )
            continue
        logger.info("Keeping %s for SHA %s", keep_path, sha)
        for entry_index, entry in items:
            if entry_index == keep_index:
                continue
            removals.append(
                DuplicateEntry(
                    entry_index=entry_index,
                    entry=entry,
                    keep_index=keep_index,
                    keep_entry=keep_entry_dict,
                )
            )
    return removals


def delete_duplicate_files(
    removals: list[DuplicateEntry],
    *,
    dry_run: bool,
    manifest_path: Path,
    create_backup: bool,
    manifest_entries: list[ManifestEntry],
) -> int:
    """Remove duplicate files and rewrite the manifest with surviving entries.

    Parameters
    ----------
    removals:
        List of duplicate manifest entries paired with the canonical copy to keep.
    dry_run:
        If true, only logs planned actions without deleting files or updating the
        manifest.
    manifest_path:
        Location of the manifest JSON file that will be rewritten.
    create_backup:
        Whether to emit a `.bak` copy of the manifest before writing changes.
    manifest_entries:
        Snapshot of the manifest contents used to compute rewrite output.

    Returns
    -------
    int
        Count of files physically deleted from disk (missing files are counted
        separately via logs).
    """
    manifest_remove_indexes: set[int] = set()
    manifest_size = len(manifest_entries)

    def mark_for_removal(entry_index: int, context: str) -> None:
        if 0 <= entry_index < manifest_size:
            manifest_remove_indexes.add(entry_index)
        else:
            logger.warning(
                "Skip manifest removal for %s; index %d exceeds snapshot size %d.",
                context,
                entry_index,
                manifest_size,
            )

    removed_file_count = 0
    missing_file_count = 0

    if dry_run and removals:
        logger.info("Dry run enabled; no files or manifest entries will be removed.")

    for duplicate in removals:
        entry = duplicate.entry
        keep_entry = duplicate.keep_entry
        entry_index = duplicate.entry_index
        sha = entry.get("sha256", "<unknown>")
        file_path = entry.get("file_path")
        keep_path = keep_entry.get("file_path", "<unknown>")

        if dry_run:
            logger.info(
                "Would remove %s (SHA %s); keeping %s",
                file_path or "<missing path>",
                sha,
                keep_path,
            )
            continue

        if file_path is None:
            logger.warning(
                "Entry for SHA %s lacks file_path. Removing manifest entry only.",
                sha,
            )
            mark_for_removal(entry_index, f"SHA {sha} missing file_path")
            continue

        if file_path == keep_path:
            logger.info(
                "Removing duplicate manifest entry for %s (SHA %s); file retained.",
                file_path,
                sha,
            )
            mark_for_removal(
                entry_index, f"SHA {sha} duplicate manifest entry {file_path}"
            )
            continue

        path = Path(file_path)
        try:
            path_exists = path.exists()
            is_symlink = path.is_symlink()
            is_broken_symlink = is_symlink and not path_exists

            if path_exists or is_broken_symlink:
                if path_exists and path.is_dir():
                    logger.warning(
                        "Skipping removal for SHA %s because %s is a directory.",
                        sha,
                        file_path,
                    )
                    mark_for_removal(
                        entry_index, f"SHA {sha} directory duplicate {file_path}"
                    )
                    continue
                path.unlink()
                removed_file_count += 1
                logger.info("Removed duplicate file %s (SHA %s)", file_path, sha)
            else:
                missing_file_count += 1
                logger.info(
                    "Duplicate file already missing at %s (SHA %s); updating manifest.",
                    file_path,
                    sha,
                )
            mark_for_removal(entry_index, f"SHA {sha} duplicate file {file_path}")
        except OSError:
            logger.exception(
                "Failed to remove duplicate file %s (SHA %s)", file_path, sha
            )

    if dry_run:
        return 0

    if manifest_remove_indexes:
        new_entries = [
            entry
            for idx, entry in enumerate(manifest_entries)
            if idx not in manifest_remove_indexes
        ]
        if create_backup:
            backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
            shutil.copy2(manifest_path, backup_path)
            logger.info("Wrote manifest backup to %s", backup_path)
        temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as tmp_manifest:
            json.dump(new_entries, tmp_manifest, indent=2)
            tmp_manifest.write("\n")
        temp_path.replace(manifest_path)
        removed_count = len(manifest_entries) - len(new_entries)
        logger.info(
            "Updated manifest: removed %d duplicate entr%s.",
            removed_count,
            "y" if removed_count == 1 else "ies",
        )

    if removed_file_count or missing_file_count:
        logger.info(
            "Duplicate pruning complete. Files removed: %d; missing on disk: %d.",
            removed_file_count,
            missing_file_count,
        )

    return removed_file_count


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level, force=True)
    logger.info("Starting duplicate pruning script")

    try:
        manifest_entries = load_manifest(args.manifest)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Unable to load manifest: %s", exc)
        return 1

    duplicate_groups = group_duplicates(manifest_entries)
    if not duplicate_groups:
        logger.info("No duplicate SHA entries found in manifest.")
        return 0

    total_duplicate_shas = len(duplicate_groups)
    total_entries = sum(len(items) for items in duplicate_groups.values())
    logger.info(
        "Identified %d duplicate SHA group%s (%d manifest entries).",
        total_duplicate_shas,
        "" if total_duplicate_shas == 1 else "s",
        total_entries,
    )

    removals = select_removals(duplicate_groups)
    if not removals:
        logger.info("No removable duplicates identified.")
        return 0

    delete_duplicate_files(
        removals,
        dry_run=args.dry_run,
        manifest_path=args.manifest,
        create_backup=args.backup,
        manifest_entries=manifest_entries,
    )

    if args.dry_run:
        logger.info("Dry run complete; no changes applied.")

    logger.info("Duplicate pruning finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
