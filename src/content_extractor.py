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


def extract_content_from_docx(file_path: str) -> str:
    """Extract all text content from a .docx file.

    Parameters
    ----------
    file_path : str
        Path to the .docx file.

    Returns
    -------
    str
        The extracted text from the document.
    """
    from docx import Document

    doc = Document(file_path)
    text_parts = [paragraph.text for paragraph in doc.paragraphs]
    return "\n".join(text_parts)


def extract_content_from_image(file_path: str) -> str:
    """Extract text from an image using OCR after preprocessing.

    Parameters
    ----------
    file_path : str
        Path to the image file.

    Returns
    -------
    str
        The extracted text from the image.
    """
    import pytesseract

    image = Image.open(file_path)
    processed_image = preprocess_for_ocr(image)
    text = pytesseract.image_to_string(processed_image)
    return text
