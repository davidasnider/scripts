from __future__ import annotations

import pytest

from app import apply_manifest_filters
from src.schema import (
    COMPLETE,
    PENDING_ANALYSIS,
    AnalysisName,
    AnalysisStatus,
)


@pytest.fixture
def base_record() -> dict:
    """Return a base file record for testing."""
    return {
        "file_path": "/tmp/example.txt",
        "file_name": "example.txt",
        "mime_type": "text/plain",
        "file_size": 1,
        "last_modified": 0.0,
        "sha256": "hash",
        "status": COMPLETE,
        "analysis_tasks": [],
    }


def test_filter_fully_analyzed_includes_completed_tasks(base_record):
    base_record["analysis_tasks"] = [
        {"name": AnalysisName.TEXT_ANALYSIS, "status": AnalysisStatus.COMPLETE}
    ]
    entries = [base_record]
    filters = {"fully_analyzed": True}
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 1
    assert result[0]["file_path"] == "/tmp/example.txt"


def test_filter_fully_analyzed_excludes_pending_tasks(base_record):
    base_record["analysis_tasks"] = [
        {"name": AnalysisName.TEXT_ANALYSIS, "status": AnalysisStatus.PENDING}
    ]
    entries = [base_record]
    filters = {"fully_analyzed": True}
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 0


def test_filter_fully_analyzed_excludes_empty_tasks(base_record):
    base_record["analysis_tasks"] = []
    entries = [base_record]
    filters = {"fully_analyzed": True}
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 0


def test_filter_by_specific_completed_task(base_record):
    base_record["analysis_tasks"] = [
        {"name": AnalysisName.TEXT_ANALYSIS, "status": AnalysisStatus.COMPLETE}
    ]
    entries = [base_record]
    filters = {"analysis_tasks": [AnalysisName.TEXT_ANALYSIS]}
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 1


def test_filter_by_specific_task_excludes_uncompleted(base_record):
    base_record["analysis_tasks"] = [
        {"name": AnalysisName.TEXT_ANALYSIS, "status": AnalysisStatus.PENDING}
    ]
    entries = [base_record]
    filters = {"analysis_tasks": [AnalysisName.TEXT_ANALYSIS]}
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 0


def test_filter_no_tasks_complete_includes_correct_files(base_record):
    base_record["status"] = COMPLETE
    base_record["analysis_tasks"] = []
    entries = [base_record]
    filters = {"no_tasks_complete": True}
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 1


def test_filter_no_tasks_complete_excludes_files_with_tasks(base_record):
    base_record["status"] = COMPLETE
    base_record["analysis_tasks"] = [
        {"name": AnalysisName.TEXT_ANALYSIS, "status": AnalysisStatus.COMPLETE}
    ]
    entries = [base_record]
    filters = {"no_tasks_complete": True}
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 0


def test_filter_no_tasks_complete_excludes_pending_files(base_record):
    base_record["status"] = PENDING_ANALYSIS
    base_record["analysis_tasks"] = []
    entries = [base_record]
    filters = {"no_tasks_complete": True}
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 0


def test_combined_filters_fully_analyzed_and_specific_task(base_record):
    record1 = base_record.copy()
    record1["file_path"] = "/tmp/file1.txt"
    record1["analysis_tasks"] = [
        {"name": AnalysisName.TEXT_ANALYSIS, "status": AnalysisStatus.COMPLETE},
        {"name": AnalysisName.PEOPLE_ANALYSIS, "status": AnalysisStatus.COMPLETE},
    ]

    record2 = base_record.copy()
    record2["file_path"] = "/tmp/file2.txt"
    record2["analysis_tasks"] = [
        {"name": AnalysisName.TEXT_ANALYSIS, "status": AnalysisStatus.COMPLETE}
    ]

    entries = [record1, record2]
    filters = {
        "fully_analyzed": True,
        "analysis_tasks": [AnalysisName.PEOPLE_ANALYSIS],
    }
    result = apply_manifest_filters(entries, filters)
    assert len(result) == 1
    assert result[0]["file_path"] == "/tmp/file1.txt"
