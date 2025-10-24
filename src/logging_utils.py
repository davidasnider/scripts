"""Shared logging configuration helpers for the file catalog project."""

from __future__ import annotations

import logging
import os
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
_CONFIGURED = False


def _resolve_level(level: int | str | None) -> int:
    """Resolve a logging level from an int, string, or environment default."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            return resolved
    env_level = os.getenv("FILE_CATALOG_LOG_LEVEL", "INFO").upper()
    resolved = logging.getLevelName(env_level)
    if isinstance(resolved, int):
        return resolved
    return logging.INFO


def configure_logging(
    level: int | str | None = None,
    *,
    log_file: str | os.PathLike[str] | None = None,
    console: bool = True,
    force: bool = False,
) -> None:
    """Configure root logging once with consistent handlers and formatting.

    Parameters
    ----------
    level:
        Optional logging level override. Falls back to FILE_CATALOG_LOG_LEVEL env.
    log_file:
        Optional path for a file handler (defaults to FILE_CATALOG_LOG_FILE env or
        ``file_catalog.log``). Pass ``None`` to disable file logging.
    console:
        Whether to attach a stream handler to stdout. Defaults to True.
    force:
        Force reconfiguration even if logging has already been configured in this
        process. Useful for CLI entrypoints that should reset previous settings.
    """
    global _CONFIGURED

    resolved_level = _resolve_level(level)
    format_string = os.getenv("FILE_CATALOG_LOG_FORMAT", DEFAULT_FORMAT)

    handlers: list[logging.Handler] = []
    if console:
        handlers.append(logging.StreamHandler())

    if log_file is None:
        log_file = os.getenv("FILE_CATALOG_LOG_FILE", "file_catalog.log")

    if log_file:
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    if not handlers:
        raise ValueError("configure_logging requires at least one handler")

    if force or not _CONFIGURED:
        logging.basicConfig(
            level=resolved_level,
            format=format_string,
            handlers=handlers,
            force=True,
        )
        _CONFIGURED = True
    else:
        root_logger = logging.getLogger()
        root_logger.setLevel(resolved_level)

    # Quiet noisy dependencies while keeping warnings/errors
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logging.getLogger("access_parser").setLevel(logging.INFO)


__all__ = ["configure_logging"]
