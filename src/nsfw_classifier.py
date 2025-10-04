"""NSFW classification using transformers."""

from __future__ import annotations

import logging
import warnings
from typing import Any

from transformers import pipeline

# Suppress transformers warnings before importing
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*slow processor.*")
warnings.filterwarnings("ignore", message=".*Device set to use.*")

logger = logging.getLogger(__name__)


class NSFWClassifier:
    """Classifier for detecting NSFW content in images using a ViT model."""

    def __init__(self) -> None:
        logger.debug("Initializing NSFW classifier")
        self.pipeline = pipeline(
            "image-classification",
            model="AdamCodd/vit-base-nsfw-detector",
        )
        logger.debug("NSFW classifier initialized")

    def classify_image(self, image_path: str) -> dict[str, Any]:
        """Classify an image for NSFW content.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        dict[str, Any]
            Classification result with 'label' and 'score'.
        """
        logger.debug("Classifying image for NSFW: %s", image_path)
        results = self.pipeline(image_path)
        # Return the top result
        logger.debug("Classification result: %s", results[0])
        return results[0]
