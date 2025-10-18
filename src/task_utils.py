from __future__ import annotations

import logging

from src.schema import ANALYSIS_TASK_VERSIONS, AnalysisName, AnalysisTask

logger = logging.getLogger("file_catalog.task_selection")


def _create_task(name: AnalysisName) -> AnalysisTask:
    """Create an analysis task with the current configured version."""

    if name not in ANALYSIS_TASK_VERSIONS:
        logger.error("Unknown analysis task name: %s", name)
        raise ValueError(f"Unknown analysis task name: {name}")
    return AnalysisTask(name=name, version=ANALYSIS_TASK_VERSIONS[name])


def _create_tasks(*names: AnalysisName) -> list[AnalysisTask]:
    return [_create_task(name) for name in names]


def determine_analysis_tasks(mime_type: str, file_path: str = "") -> list[AnalysisTask]:
    """Return the list of analysis tasks appropriate for the given MIME type."""

    tasks: list[AnalysisTask] = []
    normalized_mime = mime_type.lower()

    if normalized_mime == "application/x-msaccess":
        tasks.extend(
            _create_tasks(
                AnalysisName.ACCESS_DB_ANALYSIS,
                AnalysisName.TEXT_ANALYSIS,
                AnalysisName.PEOPLE_ANALYSIS,
            )
        )
        return tasks

    if (
        mime_type.startswith("text/")
        or mime_type == "application/pdf"
        or mime_type.endswith("document")
        or mime_type.endswith("sheet")
    ):
        tasks.extend(
            _create_tasks(
                AnalysisName.TEXT_ANALYSIS,
                AnalysisName.PEOPLE_ANALYSIS,
            )
        )
    if mime_type.startswith("image/"):
        tasks.extend(
            _create_tasks(
                AnalysisName.IMAGE_DESCRIPTION,
                AnalysisName.NSFW_CLASSIFICATION,
                AnalysisName.TEXT_ANALYSIS,
                AnalysisName.PEOPLE_ANALYSIS,
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
