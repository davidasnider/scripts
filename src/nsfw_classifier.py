"""NSFW classification using transformers."""

from __future__ import annotations

from typing import Any

from transformers import pipeline


class NSFWClassifier:
    """Classifier for detecting NSFW content in images using a ViT model."""

    def __init__(self) -> None:
        self.pipeline = pipeline(
            "image-classification",
            model="AdamCodd/vit-base-nsfw-detector",
        )

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
        results = self.pipeline(image_path)
        # Return the top result
        return results[0]
