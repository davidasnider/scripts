from __future__ import annotations

from src.manifest_utils import reset_outdated_analysis_tasks
from src.schema import (
    ANALYSIS_TASK_VERSIONS,
    COMPLETE,
    FAILED,
    PENDING_ANALYSIS,
    AnalysisName,
    AnalysisStatus,
    AnalysisTask,
    FileRecord,
)


def _build_file_record(
    status: str = COMPLETE,
    task_status: AnalysisStatus = AnalysisStatus.COMPLETE,
    task_version_delta: int = -1,
    task_name: AnalysisName = AnalysisName.TEXT_ANALYSIS,
):
    expected_version = ANALYSIS_TASK_VERSIONS[task_name]
    task_version = max(0, expected_version + task_version_delta)
    return FileRecord(
        file_path="/tmp/example.txt",
        file_name="example.txt",
        mime_type="text/plain",
        file_size=1,
        last_modified=0.0,
        sha256="hash",
        status=status,
        analysis_tasks=[
            AnalysisTask(
                name=task_name,
                status=task_status,
                version=task_version,
            )
        ],
    )


def test_reset_outdated_analysis_tasks_updates_text_analysis():
    record = _build_file_record(task_version_delta=-1, task_name=AnalysisName.TEXT_ANALYSIS)
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 1
    task = record.analysis_tasks[0]
    assert task.status == AnalysisStatus.PENDING
    assert task.version == ANALYSIS_TASK_VERSIONS[AnalysisName.TEXT_ANALYSIS]
    assert record.status == PENDING_ANALYSIS


def test_reset_outdated_analysis_tasks_updates_image_description():
    record = _build_file_record(
        task_version_delta=-1, task_name=AnalysisName.IMAGE_DESCRIPTION
    )
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 1
    task = record.analysis_tasks[0]
    assert task.status == AnalysisStatus.PENDING
    assert task.version == ANALYSIS_TASK_VERSIONS[AnalysisName.IMAGE_DESCRIPTION]
    assert record.status == PENDING_ANALYSIS


def test_reset_outdated_analysis_tasks_updates_people_analysis():
    record = _build_file_record(
        task_version_delta=-1, task_name=AnalysisName.PEOPLE_ANALYSIS
    )
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 1
    task = record.analysis_tasks[0]
    assert task.status == AnalysisStatus.PENDING
    assert task.version == ANALYSIS_TASK_VERSIONS[AnalysisName.PEOPLE_ANALYSIS]
    assert record.status == PENDING_ANALYSIS


def test_reset_outdated_analysis_tasks_skips_current_versions():
    manifest = [
        _build_file_record(task_version_delta=0, task_name=task_name)
        for task_name in (
            AnalysisName.TEXT_ANALYSIS,
            AnalysisName.IMAGE_DESCRIPTION,
            AnalysisName.PEOPLE_ANALYSIS,
        )
    ]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 0
    for record in manifest:
        assert record.analysis_tasks[0].status == AnalysisStatus.COMPLETE
        assert record.status == COMPLETE


def test_reset_outdated_analysis_tasks_updates_failed_records():
    record = _build_file_record(status=FAILED, task_status=AnalysisStatus.ERROR)
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 1
    task = record.analysis_tasks[0]
    assert task.status == AnalysisStatus.PENDING
    assert task.error_message is None
    assert record.status == PENDING_ANALYSIS
