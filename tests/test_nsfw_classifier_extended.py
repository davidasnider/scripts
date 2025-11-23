import unittest
from unittest.mock import MagicMock, patch

from PIL import UnidentifiedImageError

from src.nsfw_classifier import NSFWClassifier


class TestNSFWClassifierExtended(unittest.TestCase):
    @patch("src.nsfw_classifier._get_pipeline")
    def test_classify_image_success(self, mock_get_pipeline):
        """Test successful classification."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "sfw", "score": 0.99}]
        mock_get_pipeline.return_value = mock_pipeline

        classifier = NSFWClassifier()

        with patch("src.nsfw_classifier.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_open.return_value.__enter__.return_value = mock_image

            result = classifier.classify_image("dummy.jpg")

        self.assertEqual(result["label"], "sfw")
        self.assertEqual(result["score"], 0.99)

    @patch("src.nsfw_classifier._get_pipeline")
    def test_classify_image_unidentified_error(self, mock_get_pipeline):
        """Test handling of UnidentifiedImageError during open."""
        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline

        classifier = NSFWClassifier()

        with patch("src.nsfw_classifier.Image.open") as mock_open:
            mock_open.side_effect = UnidentifiedImageError("Bad image")

            result = classifier.classify_image("bad.jpg")

        self.assertEqual(result["label"], "SFW")
        self.assertEqual(result["reason"], "unsupported_image_format")

    @patch("src.nsfw_classifier._get_pipeline")
    def test_classify_image_pipeline_error(self, mock_get_pipeline):
        """Test handling of UnidentifiedImageError during pipeline execution."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = UnidentifiedImageError("Pipeline failed")
        mock_get_pipeline.return_value = mock_pipeline

        classifier = NSFWClassifier()

        with patch("src.nsfw_classifier.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_open.return_value.__enter__.return_value = mock_image

            result = classifier.classify_image("dummy.jpg")

        self.assertEqual(result["label"], "SFW")
        self.assertEqual(result["reason"], "unsupported_image_format")
