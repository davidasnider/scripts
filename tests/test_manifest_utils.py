from __future__ import annotations

from src.manifest_utils import (
    reset_file_record_for_rescan,
    reset_outdated_analysis_tasks,
)
from src.schema import (
    COMPLETE,
    AnalysisName,
    AnalysisStatus,
    AnalysisTask,
    FileRecord,
)


def _make_record(status: str = COMPLETE) -> FileRecord:
    return FileRecord(
        file_path="/tmp/demo.txt",
        file_name="demo.txt",
        mime_type="text/plain",
        file_size=10,
        last_modified=0.0,
        sha256="abc",
        status=status,
        analysis_tasks=[
            AnalysisTask(
                name=AnalysisName.TEXT_ANALYSIS,
                status=AnalysisStatus.COMPLETE,
                version=0,
            ),
            AnalysisTask(
                name=AnalysisName.PASSWORD_DETECTION,
                status=AnalysisStatus.COMPLETE,
                version=0,
            ),
            AnalysisTask(
                name=AnalysisName.ESTATE_ANALYSIS,
                status=AnalysisStatus.COMPLETE,
                version=0,
            ),
        ],
        contains_password=True,
        passwords=[{"old": "value"}],
        has_estate_relevant_info=True,
        estate_information={"Legal": [{"item": "Will"}]},
    )


def test_reset_outdated_analysis_tasks_updates_versions_and_adds_missing():
    record = _make_record()
    reset_count = reset_outdated_analysis_tasks([record])

    task_names = {task.name for task in record.analysis_tasks}
    expected_tasks = {
        AnalysisName.TEXT_ANALYSIS,
    }
    assert expected_tasks.issubset(task_names)
    assert reset_count >= 1  # at least one update occurred

    # Original task should be reset to pending with updated version
    text_task = next(
        task
        for task in record.analysis_tasks
        if task.name is AnalysisName.TEXT_ANALYSIS
    )
    assert text_task.status == AnalysisStatus.PENDING
    assert text_task.version >= 1
    # Sensitive flags cleared when tasks are reset
    assert record.contains_password is None
    assert record.passwords == []
    assert record.has_estate_relevant_info is None
    assert record.estate_information == {}
    # Record status should move back to pending analysis
    assert record.status == "pending_analysis"


def test_reset_file_record_for_rescan_clears_state():
    record = _make_record(status="error")
    reset_file_record_for_rescan(record)

    assert record.status == "pending_extraction"
    assert record.extracted_text is None
    assert not record.analysis_tasks
    assert all(task.status == AnalysisStatus.PENDING for task in record.analysis_tasks)
    assert all(task.error_message is None for task in record.analysis_tasks)
