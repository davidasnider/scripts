from pathlib import Path

from main import filter_records_by_search_term
from src.schema import FileRecord


def _make_record(path: str) -> FileRecord:
    name = Path(path).name
    return FileRecord(
        file_path=path,
        file_name=name,
        mime_type="text/plain",
        file_size=1,
        last_modified=0.0,
        sha256=f"hash-{name}",
    )


def test_filter_records_by_search_term_matches_path_substring():
    records = [
        _make_record("/tmp/reports/quarterly.txt"),
        _make_record("/tmp/notes/todo.txt"),
    ]

    matches = filter_records_by_search_term(records, "reports")

    assert records[0] in matches
    assert records[1] not in matches


def test_filter_records_by_search_term_matches_basename_case_insensitive():
    records = [
        _make_record("/tmp/reports/quarterly.txt"),
        _make_record("/tmp/reports/summary.txt"),
    ]

    matches = filter_records_by_search_term(records, "QUARTERLY.TXT")

    assert matches == [records[0]]


def test_filter_records_by_search_term_returns_empty_for_blank_input():
    records = [
        _make_record("/tmp/reports/quarterly.txt"),
    ]

    matches = filter_records_by_search_term(records, "   ")

    assert matches == []
