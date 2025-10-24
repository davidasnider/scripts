"""Helpers for working with the project configuration file."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path("config.yaml")

DEFAULT_CHUNK_RESERVE_TOKENS = 1024
DEFAULT_CHUNK_FALLBACK_TOKENS = 2048
DEFAULT_CHUNK_MIN_TOKENS = 512
DEFAULT_CHUNK_SAFETY_RATIO = 1.0


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_model_entry(entry: Any) -> dict[str, Any]:
    """Return a normalized model entry with name and context window."""
    if isinstance(entry, str):
        return {"name": entry, "context_window": None}

    if isinstance(entry, Mapping):
        if "name" not in entry:
            raise KeyError("Model configuration entries must include a 'name'.")
        normalized = dict(entry)
        normalized.setdefault("context_window", None)
        return normalized

    raise TypeError("Model configuration entries must be strings or mappings.")


def get_model_config(models_section: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Fetch and normalize a model's configuration."""
    if key not in models_section:
        raise KeyError(f"Model '{key}' not found in configuration.")

    entry = models_section[key]
    return _normalize_model_entry(entry)


def build_ollama_options(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return Ollama options derived from the model configuration."""
    options: dict[str, Any] = {}
    context_window = model_config.get("context_window")
    if context_window:
        options["num_ctx"] = context_window
    return options


def compute_chunk_size(
    context_window: int | None,
    *,
    reserve_tokens: int = DEFAULT_CHUNK_RESERVE_TOKENS,
    fallback_tokens: int = DEFAULT_CHUNK_FALLBACK_TOKENS,
    minimum_tokens: int = DEFAULT_CHUNK_MIN_TOKENS,
    safety_ratio: float = DEFAULT_CHUNK_SAFETY_RATIO,
) -> int:
    """Derive a safe chunk size from the provided context window."""
    if context_window is None:
        return max(minimum_tokens, fallback_tokens)

    available = context_window - reserve_tokens
    if available < minimum_tokens:
        base = minimum_tokens
    else:
        base = available

    if not 0 < safety_ratio <= 1:
        safety_ratio = DEFAULT_CHUNK_SAFETY_RATIO

    adjusted = int(base * safety_ratio)
    if adjusted <= 0:
        adjusted = minimum_tokens

    max_allowed = max(context_window - 1, minimum_tokens)
    return min(max(adjusted, minimum_tokens), max_allowed)
