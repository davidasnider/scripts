#!/usr/bin/env python3
"""Remove duplicate files listed in the manifest by pruning matching SHA entries."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from logging_utils import configure_logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DuplicateEntry:
    entry: dict
    keep_entry: dict


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


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}")
    with path.open("r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)
    if not isinstance(data, list):
        raise ValueError(f"Manifest at {path} must be a JSON array.")
    return data


def group_duplicates(entries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not sha:
            logger.debug("Skipping entry without sha256: %s", entry)
            continue
        groups.setdefault(sha, []).append(entry)
    return {sha: items for sha, items in groups.items() if len(items) > 1}


def select_removals(duplicate_groups: dict[str, list[dict]]) -> list[DuplicateEntry]:
    removals: list[DuplicateEntry] = []
    for sha, items in duplicate_groups.items():
        candidates = [
            entry for entry in items if isinstance(entry.get("file_path"), str)
        ]
        if not candidates:
            logger.warning(
                "SHA %s has duplicates but none include a valid file_path; skipping.",
                sha,
            )
            continue
        candidates.sort(key=lambda item: (len(item["file_path"]), item["file_path"]))
        keep_entry = candidates[0]
        keep_path = keep_entry.get("file_path", "<unknown>")
        logger.info("Keeping %s for SHA %s", keep_path, sha)
        for entry in items:
            if entry is keep_entry:
                continue
            removals.append(DuplicateEntry(entry=entry, keep_entry=keep_entry))
    return removals


def delete_duplicate_files(
    removals: list[DuplicateEntry],
    *,
    dry_run: bool,
    manifest_path: Path,
    create_backup: bool,
    manifest_entries: list[dict],
) -> int:
    manifest_remove_ids: set[int] = set()
    removed_file_count = 0
    missing_file_count = 0

    if dry_run and removals:
        logger.info("Dry run enabled; no files or manifest entries will be removed.")

    for duplicate in removals:
        entry = duplicate.entry
        keep_entry = duplicate.keep_entry
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
            manifest_remove_ids.add(id(entry))
            continue

        if file_path == keep_path:
            logger.info(
                "Removing duplicate manifest entry for %s (SHA %s); file retained.",
                file_path,
                sha,
            )
            manifest_remove_ids.add(id(entry))
            continue

        path = Path(file_path)
        try:
            if path.exists() or path.is_symlink():
                if path.is_dir():
                    logger.warning(
                        "Skipping removal for SHA %s because %s is a directory.",
                        sha,
                        file_path,
                    )
                    manifest_remove_ids.add(id(entry))
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
            manifest_remove_ids.add(id(entry))
        except OSError:
            logger.exception(
                "Failed to remove duplicate file %s (SHA %s)", file_path, sha
            )

    if dry_run:
        return 0

    if manifest_remove_ids:
        new_entries = [
            entry for entry in manifest_entries if id(entry) not in manifest_remove_ids
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
        logger.info(
            "Updated manifest: removed %d duplicate entr%s.",
            len(manifest_entries) - len(new_entries),
            "y" if len(manifest_entries) - len(new_entries) == 1 else "ies",
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
    logger = logging.getLogger("file_catalog.prune_duplicates")
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
