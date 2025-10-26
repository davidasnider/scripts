"""Utilities for extracting and preprocessing file content."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from docx import Document  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    Document = None  # type: ignore[assignment]

try:
    import pandas as pd  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def extract_text_from_svg(file_path: str) -> str:
    """Extract text content from an SVG file.

    Parameters
    ----------
    file_path : str
        Path to the SVG file.

    Returns
    -------
    str
        The extracted text from the SVG file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

        # Extract text from <text> elements
        text_elements = re.findall(
            r"<text[^>]*>(.*?)</text>", svg_content, re.DOTALL | re.IGNORECASE
        )

        # Extract text from <tspan> elements (often used within <text>)
        tspan_elements = re.findall(
            r"<tspan[^>]*>(.*?)</tspan>", svg_content, re.DOTALL | re.IGNORECASE
        )

        # Clean up extracted text (remove HTML entities, extra whitespace)
        all_text = text_elements + tspan_elements
        cleaned_text = []

        for text in all_text:
            # Remove any remaining HTML tags
            clean_text = re.sub(r"<[^>]+>", "", text)
            # Decode common HTML entities
            clean_text = (
                clean_text.replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
            )
            # Clean up whitespace
            clean_text = " ".join(clean_text.split())
            if clean_text.strip():
                cleaned_text.append(clean_text.strip())

        result = "\n".join(cleaned_text)
        logger.debug("Extracted %d characters from SVG: %s", len(result), file_path)
        return result

    except Exception as e:
        logger.error("Error extracting text from SVG %s: %s", file_path, e)
        return ""


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
    if Document is None:
        logger.error("python-docx library not available for DOCX extraction")
        return "DOCX extraction unavailable - python-docx not installed"

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
        # Check if it's an SVG file
        if file_path.lower().endswith(".svg"):
            return extract_text_from_svg(file_path)

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

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error("Failed to open PDF %s: %s", file_path, e)
        return ""

    all_text = []

    try:
        for page_num, page in enumerate(doc):
            try:
                text = page.get_text()
                if len(text.strip()) < 100:
                    # Assume scanned page, perform OCR
                    try:
                        pix = page.get_pixmap(dpi=300)
                        img_bytes = pix.tobytes()
                        img = Image.open(io.BytesIO(img_bytes))
                        processed_img = preprocess_for_ocr(img)
                        text = pytesseract.image_to_string(processed_img)
                    except Exception as ocr_error:
                        logger.warning(
                            "OCR failed for page %d in %s: %s",
                            page_num + 1,
                            file_path,
                            ocr_error,
                        )
                        # Keep the original text even if OCR fails
                all_text.append(text)
            except Exception as page_error:
                logger.warning(
                    "Error processing page %d in %s: %s",
                    page_num + 1,
                    file_path,
                    page_error,
                )
                # Continue with next page instead of failing entirely
                continue
    except Exception as e:
        logger.error("Error during PDF processing for %s: %s", file_path, e)
    finally:
        try:
            doc.close()
        except Exception:
            pass  # Ignore errors when closing

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

    if pd is None:
        logger.error("pandas library not available for Excel extraction")
        return "Excel extraction unavailable - pandas not installed"

    try:
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(file_path)
        all_text = []

        for sheet_name in excel_file.sheet_names:
            logger.debug("Processing sheet: %s", sheet_name)

            # Add sheet name as a header
            all_text.append(f"=== SHEET: {sheet_name} ===")

            try:
                # Read the sheet
                df = excel_file.parse(sheet_name)

                # Convert the dataframe to a readable text format
                # Include column headers
                if not df.empty:
                    # Add column headers
                    headers = " | ".join(str(col) for col in df.columns)
                    all_text.append(f"Columns: {headers}")

                    # Add each row
                    for row in df.itertuples(index=False, name=None):
                        row_text = " | ".join(str(val) for val in row if pd.notna(val))
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

    except Exception as e:
        logger.error("Error processing Excel file %s: %s", file_path, e)
        return f"Excel extraction failed: {e}"


def extract_content_from_xls(file_path: str) -> str:
    """Extract all text content from a legacy Excel (.xls) file.

    Parameters
    ----------
    file_path : str
        Path to the .xls file.

    Returns
    -------
    str
        The extracted text from all worksheets, including headers and data.
    """
    logger.debug("Extracting content from legacy Excel file: %s", file_path)

    if pd is None:
        logger.error("pandas library not available for Excel extraction")
        return "Excel extraction unavailable - pandas not installed"

    try:
        # Read all sheets from the Excel file using xlrd engine
        excel_file = pd.ExcelFile(file_path, engine="xlrd")
        all_text = []

        for sheet_name in excel_file.sheet_names:
            logger.debug("Processing sheet: %s", sheet_name)

            # Add sheet name as a header
            all_text.append(f"=== SHEET: {sheet_name} ===")

            try:
                # Read the sheet
                df = excel_file.parse(sheet_name)

                # Convert the dataframe to a readable text format
                # Include column headers
                if not df.empty:
                    # Add column headers
                    headers = " | ".join(str(col) for col in df.columns)
                    all_text.append(f"Columns: {headers}")

                    # Add each row
                    for row in df.itertuples(index=False, name=None):
                        row_text = " | ".join(str(val) for val in row if pd.notna(val))
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
            "Successfully extracted %d characters from legacy Excel file", len(result)
        )
        return result

    except Exception as e:
        logger.error("Error processing legacy Excel file %s: %s", file_path, e)
        return f"Legacy Excel extraction failed: {e}"


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
