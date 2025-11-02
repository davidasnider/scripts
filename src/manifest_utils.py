from __future__ import annotations

import logging

from src.schema import (
    ANALYSIS_TASK_VERSIONS,
    COMPLETE,
    FAILED,
    PENDING_ANALYSIS,
    PENDING_EXTRACTION,
    AnalysisName,
    AnalysisStatus,
    FileRecord,
)
from src.task_utils import determine_analysis_tasks

logger = logging.getLogger("file_catalog.analysis.versioning")


def reset_outdated_analysis_tasks(manifest: list[FileRecord]) -> int:
    """Reset manifest tasks whose stored version is outdated."""

    reset_count = 0

    for file_record in manifest:
        record_updated = False
        existing_tasks = {task.name: task for task in file_record.analysis_tasks}
        required_tasks = determine_analysis_tasks(
            file_record.mime_type,
            file_record.file_path,
            file_record.extracted_text is not None,
        )

        for required_task in required_tasks:
            if required_task.name not in existing_tasks:
                logger.info(
                    "Adding missing analysis task %s for %s (version %d)",
                    required_task.name.value,
                    file_record.file_path,
                    required_task.version,
                )
                file_record.analysis_tasks.append(required_task)
                existing_tasks[required_task.name] = required_task
                record_updated = True
                reset_count += 1

                if required_task.name is AnalysisName.PASSWORD_DETECTION:
                    file_record.contains_password = None
                    file_record.passwords = []
                elif required_task.name is AnalysisName.ESTATE_ANALYSIS:
                    file_record.has_estate_relevant_info = None
                    file_record.estate_information = {}

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
                if task.name is AnalysisName.PASSWORD_DETECTION:
                    file_record.contains_password = None
                    file_record.passwords = []
                elif task.name is AnalysisName.ESTATE_ANALYSIS:
                    file_record.has_estate_relevant_info = None
                    file_record.estate_information = {}
                reset_count += 1
                record_updated = True

        if record_updated and file_record.status in {COMPLETE, FAILED}:
            file_record.status = PENDING_ANALYSIS

    return reset_count


def reset_file_record_for_rescan(file_record: FileRecord) -> None:
    """Clear previous analysis state so the file is treated as freshly discovered."""

    tasks = determine_analysis_tasks(
        file_record.mime_type,
        file_record.file_path,
        file_record.extracted_text is not None,
    )

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
    file_record.contains_password = None
    file_record.passwords = []
    file_record.has_estate_relevant_info = None
    file_record.estate_information = {}

    file_record.analysis_tasks = tasks

    for task in file_record.analysis_tasks:
        task.status = AnalysisStatus.PENDING
        task.error_message = None
        if task.name in ANALYSIS_TASK_VERSIONS:
            task.version = ANALYSIS_TASK_VERSIONS[task.name]
