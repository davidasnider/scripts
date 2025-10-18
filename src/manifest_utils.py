from __future__ import annotations

import logging

from src.schema import (
    ANALYSIS_TASK_VERSIONS,
    COMPLETE,
    FAILED,
    PENDING_ANALYSIS,
    PENDING_EXTRACTION,
    AnalysisStatus,
    AnalysisTask,
    FileRecord,
)

try:
    # Prefer direct import to avoid duplicating task-selection logic
    from src.discover_files import _get_analysis_tasks as _determine_analysis_tasks
except ImportError:  # pragma: no cover - fallback during partial installs
    _determine_analysis_tasks = None

logger = logging.getLogger("file_catalog.analysis.versioning")


def reset_outdated_analysis_tasks(manifest: list[FileRecord]) -> int:
    """Reset manifest tasks whose stored version is outdated."""

    reset_count = 0

    for file_record in manifest:
        record_updated = False
        for task in file_record.analysis_tasks:
            if task.name not in ANALYSIS_TASK_VERSIONS:
                logger.warning(
                    "Unknown analysis task %s for %s; using stored version %d",
                    getattr(task.name, "value", task.name),
                    file_record.file_path,
                    task.version,
                )
                expected_version = task.version
            else:
                expected_version = ANALYSIS_TASK_VERSIONS[task.name]
            if task.version < expected_version:
                logger.info(
                    "Updating %s for %s to version %d (was %d)",
                    task.name.value,
                    file_record.file_path,
                    expected_version,
                    task.version,
                )
                task.version = expected_version
                task.status = AnalysisStatus.PENDING
                task.error_message = None
                reset_count += 1
                record_updated = True

        if record_updated and file_record.status in {COMPLETE, FAILED}:
            file_record.status = PENDING_ANALYSIS

    return reset_count


def reset_file_record_for_rescan(file_record: FileRecord) -> None:
    """Clear previous analysis state so the file is treated as freshly discovered."""

    if _determine_analysis_tasks is not None:
        tasks = _determine_analysis_tasks(file_record.mime_type, file_record.file_path)
    else:  # pragma: no cover - defensive fallback when discover_files unavailable
        tasks = [
            AnalysisTask(
                name=task.name,
                status=AnalysisStatus.PENDING,
                version=ANALYSIS_TASK_VERSIONS.get(task.name, task.version),
            )
            for task in file_record.analysis_tasks
        ]

    file_record.status = PENDING_EXTRACTION
    file_record.extracted_text = None
    file_record.extracted_frames = None
    file_record.summary = None
    file_record.description = None
    file_record.mentioned_people = []
    file_record.is_nsfw = None
    file_record.has_financial_red_flags = None
    file_record.potential_red_flags = []
    file_record.incriminating_items = []
    file_record.confidence_score = None

    file_record.analysis_tasks = tasks

    for task in file_record.analysis_tasks:
        task.status = AnalysisStatus.PENDING
        task.error_message = None
        if task.name in ANALYSIS_TASK_VERSIONS:
            task.version = ANALYSIS_TASK_VERSIONS[task.name]
