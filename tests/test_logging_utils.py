import logging

import pytest

from src import logging_utils


def test_configure_logging_creates_file_handler(tmp_path):
    log_path = tmp_path / "logs" / "run.log"
    logging_utils._CONFIGURED = False  # reset between tests

    logging_utils.configure_logging(
        level="warning",
        log_file=log_path,
        console=False,
        force=True,
    )

    logger = logging.getLogger("file_catalog.tests")
    logger.warning("coverage-check")

    contents = log_path.read_text(encoding="utf-8")
    assert "coverage-check" in contents


def test_configure_logging_requires_handler():
    logging_utils._CONFIGURED = False

    with pytest.raises(ValueError):
        logging_utils.configure_logging(
            level="info",
            log_file="",
            console=False,
            force=True,
        )
