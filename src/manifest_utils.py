from __future__ import annotations

import logging

from src.schema import (
    ANALYSIS_TASK_VERSIONS,
    COMPLETE,
    FAILED,
    PENDING_ANALYSIS,
    AnalysisStatus,
    FileRecord,
)

logger = logging.getLogger("file_catalog.analysis.versioning")


def reset_outdated_analysis_tasks(manifest: list[FileRecord]) -> int:
    """Reset manifest tasks whose stored version is outdated."""

    reset_count = 0

    for file_record in manifest:
        record_updated = False
        for task in file_record.analysis_tasks:
            expected_version = ANALYSIS_TASK_VERSIONS.get(task.name, task.version)
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
