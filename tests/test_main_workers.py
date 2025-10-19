from main import (
    MINIMUM_WORKER_TOTAL,
    NUM_ANALYSIS_WORKERS,
    NUM_DATABASE_WORKERS,
    NUM_EXTRACTION_WORKERS,
    _resolve_worker_counts,
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
