import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image

from src.content_extractor import (
    extract_content_from_docx,
    extract_content_from_image,
    extract_content_from_xlsx,
    extract_frames_from_video,
    extract_text_from_svg,
    preprocess_for_ocr,
)

# pytesseract is an optional dependency, so we handle the import error
try:
    from pytesseract import TesseractNotFoundError
except ImportError:

    class TesseractNotFoundError(Exception):
        pass


class TestContentExtractor(unittest.TestCase):
    def test_extract_text_from_svg(self):
        """Test text extraction from a sample SVG."""
        svg_content = """
        <svg>
            <text>Hello</text>
            <tspan>World</tspan>
        </svg>
        """
        with patch("builtins.open", unittest.mock.mock_open(read_data=svg_content)):
            text = extract_text_from_svg("dummy.svg")
        self.assertEqual(text, "Hello\nWorld")

    def test_preprocess_for_ocr(self):
        """Test OCR preprocessing."""
        # Create a dummy image
        img = Image.new("RGB", (10, 10), color="red")
        processed_img = preprocess_for_ocr(img)
        # Check that it's grayscale
        self.assertEqual(processed_img.mode, "L")
        # Check that it's thresholded (i.e., only black and white pixels)
        self.assertTrue(all(p in [0, 255] for p in processed_img.getdata()))

    @patch("src.content_extractor.Image.open")
    def test_extract_content_from_image(self, mock_image_open):
        """Test text extraction from a generic image."""
        # Configure the mock image to have the correct data type
        mock_image = MagicMock()
        mock_image.mode = "L"
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image

        # Correct the dtype of the numpy array
        mock_array = np.zeros((10, 10), dtype=np.uint8)
        mock_image.load.return_value = mock_array

        # Mock np.array to return the correct array
        with patch("src.content_extractor.np.array", return_value=mock_array):
            mock_pytesseract = MagicMock()
            mock_pytesseract.image_to_string.return_value = "some text"

            with patch.dict(sys.modules, {"pytesseract": mock_pytesseract}):
                text = extract_content_from_image("dummy.png")

            self.assertEqual(text, "some text")

    @patch("src.content_extractor.Image.open")
    def test_extract_content_from_image_tesseract_not_found(self, mock_image_open):
        """Test TesseractNotFoundError handling."""
        # Configure the mock image to have the correct data type
        mock_image = MagicMock()
        mock_image.mode = "L"
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image

        # Correct the dtype of the numpy array
        mock_array = np.zeros((10, 10), dtype=np.uint8)
        mock_image.load.return_value = mock_array

        # Mock np.array to return the correct array
        with patch("src.content_extractor.np.array", return_value=mock_array):
            mock_pytesseract = MagicMock()
            mock_pytesseract.TesseractNotFoundError = TesseractNotFoundError
            mock_pytesseract.image_to_string.side_effect = TesseractNotFoundError

            with patch.dict(sys.modules, {"pytesseract": mock_pytesseract}):
                text = extract_content_from_image("dummy.png")

            self.assertEqual(text, "")

    @patch("src.content_extractor.extract_text_from_svg")
    def test_extract_content_from_svg_image(self, mock_extract_text_from_svg):
        """Test that SVG images are routed to the correct function."""
        mock_extract_text_from_svg.return_value = "svg text"
        text = extract_content_from_image("dummy.svg")
        self.assertEqual(text, "svg text")

    def test_extract_content_from_docx(self):
        """Test text extraction from a .docx file."""
        # Create a mock Document object
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(), MagicMock()]
        mock_doc.paragraphs[0].text = "Hello"
        mock_doc.paragraphs[1].text = "World"

        with patch("src.content_extractor.Document", return_value=mock_doc):
            text = extract_content_from_docx("dummy.docx")

        self.assertEqual(text, "Hello\nWorld")

    def test_extract_content_from_xlsx(self):
        """Test text extraction from an .xlsx file."""
        # Create a mock Excel file
        mock_excel_file = MagicMock()
        mock_excel_file.sheet_names = ["Sheet1"]

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_excel_file.parse.return_value = mock_df

        with patch(
            "src.content_extractor.pd.ExcelFile",
            return_value=mock_excel_file,
        ):
            text = extract_content_from_xlsx("dummy.xlsx")

        self.assertIn("SHEET: Sheet1", text)
        self.assertIn("1 | a", text)
        self.assertIn("2 | b", text)

    @patch("src.content_extractor.cv2.VideoCapture")
    @patch("src.content_extractor.cv2.imwrite")
    def test_extract_frames_from_video(self, mock_imwrite, mock_videocapture):
        """Test frame extraction from a video."""
        # Mock the video capture object
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30  # fps
        # Simulate 3 frames
        mock_cap.read.side_effect = [
            (True, "frame1"),
            (True, "frame2"),
            (True, "frame3"),
            (False, None),
        ]
        mock_videocapture.return_value = mock_cap

        frames = extract_frames_from_video("dummy.mp4", "output", 1)

        self.assertEqual(len(frames), 1)
        self.assertIn("frame_000000.jpg", frames[0])
