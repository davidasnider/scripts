from src.schema import PENDING_EXTRACTION, FileRecord


def test_file_record_defaults():
    """
    Verify that a FileRecord can be created with minimal data and defaults are set.
    """
    record = FileRecord(
        file_path="/path/to/file.txt",
        file_name="file.txt",
        mime_type="text/plain",
        file_size=123,
        last_modified=1678886400.0,
        sha256="a1b2c3d4",
    )
    assert record.status == PENDING_EXTRACTION
    assert record.extracted_text is None
    assert record.extracted_frames is None
    assert record.analysis_tasks == []
    assert record.summary is None
    assert record.description is None
    assert record.mentioned_people == []
    assert record.is_nsfw is None
    assert record.has_financial_red_flags is None
    assert record.potential_red_flags == []
    assert record.incriminating_items == []
    assert record.confidence_score is None


def test_file_record_summary_coerces_structured_data():
    record = FileRecord.model_validate(  # type: ignore[arg-type]
        {
            "file_path": "/path/to/file.txt",
            "file_name": "file.txt",
            "mime_type": "text/plain",
            "file_size": 123,
            "last_modified": 1678886400.0,
            "sha256": "a1b2c3d4",
            "summary": {"name": "John", "favorite_food": "Pizza"},
        }
    )
    assert record.summary == '{"favorite_food": "Pizza", "name": "John"}'
