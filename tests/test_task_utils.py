from __future__ import annotations

from src.schema import AnalysisName, AnalysisStatus, AnalysisTask, FileRecord
from src.task_utils import determine_analysis_tasks, ensure_required_tasks


def _make_record(mime: str, path: str = "/tmp/file") -> FileRecord:
    return FileRecord(
        file_path=path,
        file_name="file",
        mime_type=mime,
        file_size=1,
        last_modified=0.0,
        sha256="hash",
        analysis_tasks=[],
    )


def test_determine_analysis_tasks_handles_various_formats():
    text_tasks = determine_analysis_tasks("text/plain")
    text_names = {task.name for task in text_tasks}
    assert not text_names

    image_tasks = determine_analysis_tasks("image/png")
    image_names = {task.name for task in image_tasks}
    assert AnalysisName.IMAGE_DESCRIPTION in image_names
    assert AnalysisName.NSFW_CLASSIFICATION in image_names

    video_tasks = determine_analysis_tasks("video/mp4", "/tmp/video.mp4")
    video_names = {task.name for task in video_tasks}
    assert AnalysisName.VIDEO_SUMMARY in video_names

    access_tasks = determine_analysis_tasks("application/x-msaccess")
    access_names = [task.name for task in access_tasks]
    assert access_names[0] == AnalysisName.ACCESS_DB_ANALYSIS


def test_ensure_required_tasks_adds_missing_entries():
    record = _make_record("text/plain")
    record.extracted_text = "This is a test"
    preexisting = AnalysisTask(
        name=AnalysisName.TEXT_ANALYSIS, status=AnalysisStatus.COMPLETE
    )
    record.analysis_tasks.append(preexisting)

    ensure_required_tasks(record)

    names = {task.name for task in record.analysis_tasks}
    assert AnalysisName.TEXT_ANALYSIS in names
    assert AnalysisName.PASSWORD_DETECTION in names
    assert AnalysisName.ESTATE_ANALYSIS in names
    # Existing task status is preserved, new tasks default to pending
    assert preexisting.status == AnalysisStatus.COMPLETE
    new_tasks = [task for task in record.analysis_tasks if task is not preexisting]
    assert all(task.status == AnalysisStatus.PENDING for task in new_tasks)
