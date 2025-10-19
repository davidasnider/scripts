from __future__ import annotations

from src.manifest_utils import (
    reset_file_record_for_rescan,
    reset_outdated_analysis_tasks,
)
from src.schema import (
    ANALYSIS_TASK_VERSIONS,
    COMPLETE,
    FAILED,
    PENDING_ANALYSIS,
    PENDING_EXTRACTION,
    AnalysisName,
    AnalysisStatus,
    AnalysisTask,
    FileRecord,
)
from src.task_utils import determine_analysis_tasks


def _build_file_record(
    status: str = COMPLETE,
    task_status: AnalysisStatus = AnalysisStatus.COMPLETE,
    task_version_delta: int = -1,
    task_name: AnalysisName = AnalysisName.TEXT_ANALYSIS,
):
    expected_version = ANALYSIS_TASK_VERSIONS[task_name]
    task_version = max(0, expected_version + task_version_delta)
    mime_map = {
        AnalysisName.IMAGE_DESCRIPTION: "image/jpeg",
        AnalysisName.NSFW_CLASSIFICATION: "image/jpeg",
        AnalysisName.VIDEO_SUMMARY: "video/mp4",
        AnalysisName.ACCESS_DB_ANALYSIS: "application/x-msaccess",
    }
    mime_type = mime_map.get(task_name, "text/plain")
    return FileRecord(
        file_path="/tmp/example.txt",
        file_name="example.txt",
        mime_type=mime_type,
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
    record = _build_file_record(
        task_version_delta=-1, task_name=AnalysisName.TEXT_ANALYSIS
    )
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 4
    task_lookup = {task.name: task for task in record.analysis_tasks}
    assert task_lookup[AnalysisName.TEXT_ANALYSIS].status == AnalysisStatus.PENDING
    assert (
        task_lookup[AnalysisName.TEXT_ANALYSIS].version
        == ANALYSIS_TASK_VERSIONS[AnalysisName.TEXT_ANALYSIS]
    )
    assert {
        AnalysisName.TEXT_ANALYSIS,
        AnalysisName.PEOPLE_ANALYSIS,
        AnalysisName.ESTATE_ANALYSIS,
        AnalysisName.PASSWORD_DETECTION,
    } == set(task_lookup)
    assert record.status == PENDING_ANALYSIS


def test_reset_outdated_analysis_tasks_updates_image_description():
    record = _build_file_record(
        task_version_delta=-1, task_name=AnalysisName.IMAGE_DESCRIPTION
    )
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 6
    task_lookup = {task.name: task for task in record.analysis_tasks}
    assert task_lookup[AnalysisName.IMAGE_DESCRIPTION].status == AnalysisStatus.PENDING
    assert (
        task_lookup[AnalysisName.IMAGE_DESCRIPTION].version
        == ANALYSIS_TASK_VERSIONS[AnalysisName.IMAGE_DESCRIPTION]
    )
    assert {
        AnalysisName.IMAGE_DESCRIPTION,
        AnalysisName.NSFW_CLASSIFICATION,
        AnalysisName.TEXT_ANALYSIS,
        AnalysisName.PEOPLE_ANALYSIS,
        AnalysisName.ESTATE_ANALYSIS,
        AnalysisName.PASSWORD_DETECTION,
    } == set(task_lookup)
    assert record.status == PENDING_ANALYSIS


def test_reset_outdated_analysis_tasks_updates_people_analysis():
    record = _build_file_record(
        task_version_delta=-1, task_name=AnalysisName.PEOPLE_ANALYSIS
    )
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 4
    task_lookup = {task.name: task for task in record.analysis_tasks}
    assert task_lookup[AnalysisName.PEOPLE_ANALYSIS].status == AnalysisStatus.PENDING
    assert (
        task_lookup[AnalysisName.PEOPLE_ANALYSIS].version
        == ANALYSIS_TASK_VERSIONS[AnalysisName.PEOPLE_ANALYSIS]
    )
    assert {
        AnalysisName.TEXT_ANALYSIS,
        AnalysisName.PEOPLE_ANALYSIS,
        AnalysisName.ESTATE_ANALYSIS,
        AnalysisName.PASSWORD_DETECTION,
    } == set(task_lookup)
    assert record.status == PENDING_ANALYSIS


def test_reset_outdated_analysis_tasks_skips_text_analysis_current_version():
    path = "/tmp/text.txt"
    mime_type = "text/plain"
    tasks = determine_analysis_tasks(mime_type, path)
    for task in tasks:
        task.status = AnalysisStatus.COMPLETE
    record = FileRecord(
        file_path=path,
        file_name=path.split("/")[-1],
        mime_type=mime_type,
        file_size=1,
        last_modified=0.0,
        sha256="hash",
        status=COMPLETE,
        analysis_tasks=tasks,
    )
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 0
    assert all(task.status == AnalysisStatus.COMPLETE for task in record.analysis_tasks)
    assert record.status == COMPLETE


def test_reset_outdated_analysis_tasks_skips_image_description_current_version():
    path = "/tmp/image.jpg"
    mime_type = "image/jpeg"
    tasks = determine_analysis_tasks(mime_type, path)
    for task in tasks:
        task.status = AnalysisStatus.COMPLETE
    record = FileRecord(
        file_path=path,
        file_name=path.split("/")[-1],
        mime_type=mime_type,
        file_size=1,
        last_modified=0.0,
        sha256="hash",
        status=COMPLETE,
        analysis_tasks=tasks,
    )
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 0
    assert all(task.status == AnalysisStatus.COMPLETE for task in record.analysis_tasks)
    assert record.status == COMPLETE


def test_reset_outdated_analysis_tasks_skips_password_detection_current_version():
    path = "/tmp/password.txt"
    mime_type = "text/plain"
    tasks = determine_analysis_tasks(mime_type, path)
    for task in tasks:
        task.status = AnalysisStatus.COMPLETE
    record = FileRecord(
        file_path=path,
        file_name=path.split("/")[-1],
        mime_type=mime_type,
        file_size=1,
        last_modified=0.0,
        sha256="hash",
        status=COMPLETE,
        analysis_tasks=tasks,
    )
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 0
    assert all(task.status == AnalysisStatus.COMPLETE for task in record.analysis_tasks)
    assert record.status == COMPLETE


def test_reset_outdated_analysis_tasks_updates_failed_records():
    record = _build_file_record(status=FAILED, task_status=AnalysisStatus.ERROR)
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 4
    for task in record.analysis_tasks:
        assert task.status == AnalysisStatus.PENDING
        assert task.error_message is None
    assert record.status == PENDING_ANALYSIS
    assert record.contains_password is None
    assert record.passwords == {}


def test_reset_outdated_analysis_tasks_resets_password_fields():
    record = _build_file_record(task_name=AnalysisName.PASSWORD_DETECTION)
    record.contains_password = True
    record.passwords = {"admin": "secret"}
    manifest = [record]

    reset_count = reset_outdated_analysis_tasks(manifest)

    assert reset_count == 4
    task_lookup = {task.name: task for task in record.analysis_tasks}
    assert {
        AnalysisName.TEXT_ANALYSIS,
        AnalysisName.PEOPLE_ANALYSIS,
        AnalysisName.ESTATE_ANALYSIS,
        AnalysisName.PASSWORD_DETECTION,
    } == set(task_lookup)
    assert task_lookup[AnalysisName.PASSWORD_DETECTION].status == AnalysisStatus.PENDING
    assert record.contains_password is None
    assert record.passwords == {}


def test_reset_file_record_for_rescan_clears_previous_analysis():
    record = FileRecord(
        file_path="/tmp/report.txt",
        file_name="report.txt",
        mime_type="text/plain",
        file_size=10,
        last_modified=0.0,
        sha256="123",
        status=COMPLETE,
        extracted_text="old text",
        summary="old summary",
        description="old description",
        mentioned_people=["Alice"],
        is_nsfw=True,
        has_financial_red_flags=True,
        potential_red_flags=["flag"],
        incriminating_items=["item"],
        confidence_score=42,
        analysis_tasks=[
            AnalysisTask(
                name=AnalysisName.TEXT_ANALYSIS,
                status=AnalysisStatus.COMPLETE,
                version=ANALYSIS_TASK_VERSIONS[AnalysisName.TEXT_ANALYSIS],
            )
        ],
    )

    reset_file_record_for_rescan(record)

    assert record.status == PENDING_EXTRACTION
    assert record.extracted_text is None
    assert record.summary is None
    assert record.description is None
    assert record.mentioned_people == []
    assert record.is_nsfw is None
    assert record.has_financial_red_flags is None
    assert record.potential_red_flags == []
    assert record.incriminating_items == []
    assert record.confidence_score is None
    task_names = [task.name for task in record.analysis_tasks]
    assert task_names == [
        AnalysisName.TEXT_ANALYSIS,
        AnalysisName.PEOPLE_ANALYSIS,
        AnalysisName.ESTATE_ANALYSIS,
        AnalysisName.PASSWORD_DETECTION,
    ]
    assert all(task.status == AnalysisStatus.PENDING for task in record.analysis_tasks)


def test_reset_file_record_for_rescan_drops_access_analysis_tasks():
    record = FileRecord(
        file_path="/tmp/db.mdb",
        file_name="db.mdb",
        mime_type="application/x-msaccess",
        file_size=10,
        last_modified=0.0,
        sha256="456",
        status=COMPLETE,
        analysis_tasks=[
            AnalysisTask(
                name=AnalysisName.ACCESS_DB_ANALYSIS,
                status=AnalysisStatus.COMPLETE,
            )
        ],
    )

    reset_file_record_for_rescan(record)

    assert record.status == PENDING_EXTRACTION
    task_names = [task.name for task in record.analysis_tasks]
    assert task_names == [
        AnalysisName.ACCESS_DB_ANALYSIS,
        AnalysisName.TEXT_ANALYSIS,
        AnalysisName.PEOPLE_ANALYSIS,
        AnalysisName.ESTATE_ANALYSIS,
        AnalysisName.PASSWORD_DETECTION,
    ]
    assert all(task.status == AnalysisStatus.PENDING for task in record.analysis_tasks)
