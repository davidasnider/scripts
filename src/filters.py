from __future__ import annotations

from typing import Any

from src.schema import COMPLETE, AnalysisName, AnalysisStatus


def apply_manifest_filters(
    entries: list[dict[str, Any]], filter_state: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply filters from the inline panel to manifest rows."""
    filtered: list[dict[str, Any]] = []
    file_types = filter_state.get("file_type") or []
    hide_nsfw = filter_state.get("hide_nsfw", False)
    only_red_flags = filter_state.get("red_flags", False)
    fully_analyzed = filter_state.get("fully_analyzed", False)
    analysis_tasks_filter = filter_state.get("analysis_tasks", [])
    no_tasks_complete = filter_state.get("no_tasks_complete", False)

    def _status_value(raw_status: Any) -> str:
        if isinstance(raw_status, AnalysisStatus):
            return raw_status.value
        if raw_status is None:
            return ""
        return str(raw_status)

    def _task_name_value(raw_name: Any) -> str:
        if isinstance(raw_name, AnalysisName):
            return raw_name.value
        if raw_name is None:
            return ""
        return str(raw_name)

    required_task_names = {_task_name_value(name) for name in analysis_tasks_filter}

    for entry in entries:
        if hide_nsfw and entry.get("is_nsfw"):
            continue
        if file_types and entry.get("mime_type") not in file_types:
            continue
        if only_red_flags and not entry.get("has_financial_red_flags"):
            continue

        tasks = entry.get("analysis_tasks", [])
        if fully_analyzed:
            if not tasks or any(
                _status_value(task.get("status")) != AnalysisStatus.COMPLETE.value
                for task in tasks
            ):
                continue

        if required_task_names:
            completed_tasks = {
                _task_name_value(task.get("name"))
                for task in tasks
                if _status_value(task.get("status")) == AnalysisStatus.COMPLETE.value
            }
            if not required_task_names.issubset(completed_tasks):
                continue

        if no_tasks_complete:
            if not (_status_value(entry.get("status")) == COMPLETE and not tasks):
                continue

        filtered.append(entry)

    return filtered
