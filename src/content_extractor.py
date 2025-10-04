"""Utilities for extracting and preprocessing file content."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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
    logger.debug("Extracting content from image: %s", file_path)
    import pytesseract

    try:
        image = Image.open(file_path)
        processed_image = preprocess_for_ocr(image)
        text = pytesseract.image_to_string(processed_image)
        logger.debug("Successfully extracted %d characters from image", len(text))
        return text
    except pytesseract.TesseractNotFoundError:
        logger.warning("Tesseract not found. Skipping OCR for %s", file_path)
        return ""
    except Exception as e:
        logger.error("Error processing image %s: %s", file_path, e)
        return ""


def extract_content_from_pdf(file_path: str) -> str:
    """Extract text from a PDF using a hybrid digital/OCR strategy.

    For each page, attempts digital text extraction first. If the extracted
    text is short (<100 characters), assumes it's a scanned page and performs
    OCR after converting to a high-DPI image.

    Parameters
    ----------
    file_path : str
        Path to the PDF file.

    Returns
    -------
    str
        The extracted text from all pages.
    """
    import fitz  # PyMuPDF
    import pytesseract

    doc = fitz.open(file_path)
    all_text = []

    try:
        for page in doc:
            text = page.get_text()
            if len(text.strip()) < 100:
                # Assume scanned page, perform OCR
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes()
                img = Image.open(io.BytesIO(img_bytes))
                processed_img = preprocess_for_ocr(img)
                text = pytesseract.image_to_string(processed_img)
            all_text.append(text)
    finally:
        doc.close()

    return "\n".join(all_text)


def extract_frames_from_video(
    file_path: str, output_dir: str, interval_sec: float
) -> list[str]:
    """Extract frames from a video at regular intervals and save as JPEG images.

    Parameters
    ----------
    file_path : str
        Path to the video file.
    output_dir : str
        Directory to save the extracted frames.
    interval_sec : float
        Interval in seconds between extracted frames.

    Returns
    -------
    list[str]
        List of file paths to the saved frame images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Video has invalid FPS")

    frame_interval = int(fps * interval_sec)
    if frame_interval < 1:
        frame_interval = 1

    saved_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = output_path / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(str(frame_path))

        frame_count += 1

    cap.release()
    return saved_frames
