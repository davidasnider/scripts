from __future__ import annotations

import logging

from src.schema import (
    ANALYSIS_TASK_VERSIONS,
    AnalysisName,
    AnalysisStatus,
    AnalysisTask,
    FileRecord,
)

logger = logging.getLogger("file_catalog.task_selection")


def _create_task(name: AnalysisName) -> AnalysisTask:
    """Create an analysis task with the current configured version."""

    if name not in ANALYSIS_TASK_VERSIONS:
        logger.error("Unknown analysis task name: %s", name)
        raise ValueError(f"Unknown analysis task name: {name}")
    return AnalysisTask(name=name, version=ANALYSIS_TASK_VERSIONS[name])


def _create_tasks(*names: AnalysisName) -> list[AnalysisTask]:
    return [_create_task(name) for name in names]


def determine_analysis_tasks(
    mime_type: str, file_path: str = "", has_extracted_text: bool = False
) -> list[AnalysisTask]:
    """Return the list of analysis tasks appropriate for the given MIME type."""

    tasks: list[AnalysisTask] = []
    normalized_mime = mime_type.lower()

    if has_extracted_text:
        tasks.extend(
            _create_tasks(
                AnalysisName.TEXT_ANALYSIS,
                AnalysisName.PEOPLE_ANALYSIS,
                AnalysisName.ESTATE_ANALYSIS,
                AnalysisName.PASSWORD_DETECTION,
                AnalysisName.FINANCIAL_ANALYSIS,
            )
        )

    if normalized_mime == "application/x-msaccess":
        tasks.extend(
            _create_tasks(
                AnalysisName.ACCESS_DB_ANALYSIS,
            )
        )
        return tasks

    if mime_type.startswith("image/"):
        tasks.extend(
            _create_tasks(
                AnalysisName.IMAGE_DESCRIPTION,
                AnalysisName.NSFW_CLASSIFICATION,
            )
        )
    if mime_type.startswith("video/") and not file_path.lower().endswith(".asx"):
        tasks.extend(
            _create_tasks(
                AnalysisName.VIDEO_SUMMARY,
                AnalysisName.NSFW_CLASSIFICATION,
            )
        )

    return tasks


def ensure_required_tasks(file_record: FileRecord) -> None:
    """Add any missing analysis tasks to the record based on its MIME type."""

    existing = {task.name for task in file_record.analysis_tasks}
    required = determine_analysis_tasks(
        file_record.mime_type,
        file_record.file_path,
        has_extracted_text=bool(file_record.extracted_text),
    )

    for task in required:
        if task.name not in existing:
            task.status = AnalysisStatus.PENDING
            task.error_message = None
            file_record.analysis_tasks.append(task)
