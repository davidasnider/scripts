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


def extract_content_from_xlsx(file_path: str) -> str:
    """Extract all text content from an Excel (.xlsx) file.

    Parameters
    ----------
    file_path : str
        Path to the .xlsx file.

    Returns
    -------
    str
        The extracted text from all worksheets, including headers and data.
    """
    logger.debug("Extracting content from Excel file: %s", file_path)

    try:
        import pandas as pd

        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(file_path)
        all_text = []

        for sheet_name in excel_file.sheet_names:
            logger.debug("Processing sheet: %s", sheet_name)

            # Add sheet name as a header
            all_text.append(f"=== SHEET: {sheet_name} ===")

            try:
                # Read the sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # Convert the dataframe to a readable text format
                # Include column headers
                if not df.empty:
                    # Add column headers
                    headers = " | ".join(str(col) for col in df.columns)
                    all_text.append(f"Columns: {headers}")

                    # Add each row
                    for index, row in df.iterrows():
                        row_text = " | ".join(
                            str(val) for val in row.values if pd.notna(val)
                        )
                        if row_text.strip():  # Only add non-empty rows
                            all_text.append(row_text)
                else:
                    all_text.append("(Empty sheet)")

            except Exception as e:
                logger.warning("Error reading sheet %s: %s", sheet_name, e)
                all_text.append(f"(Error reading sheet: {e})")

            all_text.append("")  # Add blank line between sheets

        result = "\n".join(all_text)
        logger.debug(
            "Successfully extracted %d characters from Excel file", len(result)
        )
        return result

    except ImportError:
        logger.error("pandas library not available for Excel extraction")
        return "Excel extraction unavailable - pandas not installed"
    except Exception as e:
        logger.error("Error processing Excel file %s: %s", file_path, e)
        return f"Excel extraction failed: {e}"


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
