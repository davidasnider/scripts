"""NSFW classification using transformers."""

from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

# Suppress transformers warnings before importing
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*slow processor.*")
warnings.filterwarnings("ignore", message=".*Device set to use.*")

logger = logging.getLogger(__name__)

_PIPELINE_LOCK = threading.Lock()
_PIPELINE_INSTANCE = None


def _load_config() -> dict[str, Any]:
    with Path("config.yaml").open("r") as handle:
        return yaml.safe_load(handle)


def _pipeline_device() -> tuple[Any, dict[str, Any]]:
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": False}

    if torch.cuda.is_available():
        model_kwargs.setdefault("torch_dtype", torch.float32)
        return 0, model_kwargs

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_available:
        model_kwargs.setdefault("torch_dtype", torch.float32)
        return torch.device("mps"), model_kwargs

    model_kwargs["torch_dtype"] = torch.float32
    return -1, model_kwargs


def _get_pipeline():
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is not None:
        return _PIPELINE_INSTANCE
    with _PIPELINE_LOCK:
        if _PIPELINE_INSTANCE is not None:
            return _PIPELINE_INSTANCE
        config = _load_config()
        device, model_kwargs = _pipeline_device()
        _PIPELINE_INSTANCE = pipeline(
            "image-classification",
            model=config["models"]["nsfw_detector"],
            device=device,
            model_kwargs=model_kwargs,
        )
        logger.debug("NSFW pipeline prepared on device %s", device)
        return _PIPELINE_INSTANCE


class NSFWClassifier:
    """Classifier for detecting NSFW content in images using a ViT model."""

    def __init__(self) -> None:
        self.pipeline = _get_pipeline()

    def classify_image(self, image_path: str) -> dict[str, Any]:
        """Classify an image for NSFW content."""
        logger.debug("Classifying image for NSFW: %s", image_path)
        try:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
        except UnidentifiedImageError as exc:
            logger.info(
                "Skipping NSFW classification for unsupported image %s: %s",
                image_path,
                exc,
            )
            return {
                "label": "SFW",
                "score": 0.0,
                "reason": "unsupported_image_format",
            }
        except Exception:
            logger.debug("Falling back to raw path for %s", image_path)
            image = image_path
        try:
            results = self.pipeline(image)
        except UnidentifiedImageError as exc:
            logger.info(
                "Skipping NSFW classification via pipeline for "
                "unsupported image %s: %s",
                image_path,
                exc,
            )
            return {
                "label": "SFW",
                "score": 0.0,
                "reason": "unsupported_image_format",
            }
        logger.debug("Classification result: %s", results[0])
        return results[0]
