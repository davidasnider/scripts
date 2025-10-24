from main import (
    ActiveFileStatus,
    MINIMUM_WORKER_TOTAL,
    NUM_ANALYSIS_WORKERS,
    NUM_DATABASE_WORKERS,
    NUM_EXTRACTION_WORKERS,
    _apply_chunk_progress_from_log,
    _resolve_worker_counts,
    in_progress_files,
    lock,
)


def test_resolve_worker_counts_default():
    extraction, analysis, database, limit, increased = _resolve_worker_counts(None)

    assert extraction == NUM_EXTRACTION_WORKERS
    assert analysis == NUM_ANALYSIS_WORKERS
    assert database == NUM_DATABASE_WORKERS
    assert limit is None
    assert increased is False


def test_resolve_worker_counts_applies_limit():
    extraction, analysis, database, limit, increased = _resolve_worker_counts(5)

    assert (extraction, analysis, database) == (3, 1, 1)
    assert limit == 5
    assert increased is False
    assert extraction + analysis + database == 5


def test_resolve_worker_counts_enforces_minimum():
    extraction, analysis, database, limit, increased = _resolve_worker_counts(1)

    assert (extraction, analysis, database) == (1, 1, 1)
    assert limit == MINIMUM_WORKER_TOTAL
    assert increased is True


def test_apply_chunk_progress_updates_active_status():
    message = (
        "Estate analysis chunk 2/7 for AT&T QuickStart Guide.pdf (5 remaining)"
    )

    with lock:
        in_progress_files.clear()
        in_progress_files["file-abc"] = ActiveFileStatus(
            file_name="AT&T QuickStart Guide.pdf",
            stage="Pending",
        )

    try:
        updated = _apply_chunk_progress_from_log(message)
        assert updated is True
        with lock:
            status = in_progress_files["file-abc"]
            assert status.stage == "Analyzing"
            assert status.current_task == "Estate Analysis"
            assert status.chunks_total == 7
            assert status.chunks_processed == 2
    finally:
        with lock:
            in_progress_files.clear()


def test_apply_chunk_progress_resets_processed_count():
    message = "Text analysis chunk 1/9 for Report.pdf (8 remaining)"

    with lock:
        in_progress_files.clear()
        in_progress_files["file-xyz"] = ActiveFileStatus(
            file_name="Report.pdf",
            stage="Analyzing",
            current_task="Text Analysis",
            chunks_processed=4,
            chunks_total=7,
        )

    try:
        updated = _apply_chunk_progress_from_log(message)
        assert updated is True
        with lock:
            status = in_progress_files["file-xyz"]
            assert status.current_task == "Text Analysis"
            assert status.chunks_total == 9
            assert status.chunks_processed == 1
    finally:
        with lock:
            in_progress_files.clear()


def test_apply_chunk_progress_ignores_unmatched_messages():
    with lock:
        in_progress_files.clear()

    try:
        updated = _apply_chunk_progress_from_log("Unrelated log line")
        assert updated is False
    finally:
        with lock:
            in_progress_files.clear()
