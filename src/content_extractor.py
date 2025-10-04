"""Utilities for extracting and preprocessing file content."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def preprocess_for_ocr(image: Image.Image, threshold: int = 180) -> Image.Image:
    """Convert an image to grayscale and apply binary thresholding for OCR.

    Parameters
    ----------
    image:
        The input PIL image to preprocess.
    threshold:
        Threshold value (0-255) applied after grayscale conversion. Defaults to 180.

    Returns
    -------
    Image.Image
        A PIL image converted to grayscale and thresholded for OCR pipelines.
    """

    if image.mode != "L":
        grayscale = image.convert("L")
    else:
        grayscale = image

    grayscale_array = np.array(grayscale)

    _, binary_array = cv2.threshold(
        grayscale_array, threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    return Image.fromarray(binary_array)
