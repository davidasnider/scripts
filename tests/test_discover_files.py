import hashlib
import json
import logging
import mimetypes
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src import discover_files
from src.schema import AnalysisName


@pytest.fixture
def temp_directory_with_files():
    """Create a temporary directory with a predictable structure for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    (temp_dir / "subdir").mkdir()

    files = {
        "file1.txt": "This is a text file.",
        "file2.jpg": "dummy image data",
        "file3.pdf": "dummy pdf data",
        "archive.zip": "dummy zip data",
        (Path("subdir") / "file4.mov"): "dummy video data",
    }

    for rel_path, content in files.items():
        path = temp_dir / rel_path
        with open(path, "w") as f:
            f.write(content)

    yield temp_dir

    shutil.rmtree(temp_dir)


def test_calculate_sha256(temp_directory_with_files):
    """Verify that the SHA256 hash is calculated correctly."""
    test_file = temp_directory_with_files / "file1.txt"
    expected_hash = hashlib.sha256(b"This is a text file.").hexdigest()
    assert discover_files._calculate_sha256(test_file) == expected_hash


@pytest.mark.parametrize(
    "mime_type, file_path, expected_tasks",
    [
        ("text/plain", "file.txt", [AnalysisName.TEXT_ANALYSIS]),
        ("application/pdf", "file.pdf", [AnalysisName.TEXT_ANALYSIS]),
        (
            "image/jpeg",
            "image.jpg",
            [
                AnalysisName.IMAGE_DESCRIPTION,
                AnalysisName.NSFW_CLASSIFICATION,
                AnalysisName.TEXT_ANALYSIS,
            ],
        ),
        (
            "video/mp4",
            "video.mp4",
            [AnalysisName.VIDEO_SUMMARY, AnalysisName.NSFW_CLASSIFICATION],
        ),
        ("application/zip", "archive.zip", []),
        ("video/x-ms-asf", "playlist.asx", []),
        (
            "video/quicktime",
            "video.mov",
            [AnalysisName.VIDEO_SUMMARY, AnalysisName.NSFW_CLASSIFICATION],
        ),
    ],
)
def test_get_analysis_tasks(mime_type, file_path, expected_tasks):
    """Verify that analysis tasks are correctly determined from MIME types."""
    tasks = discover_files._get_analysis_tasks(mime_type, file_path)
    assert [task.name for task in tasks] == expected_tasks


def test_count_files(temp_directory_with_files):
    """Verify that the file counter correctly counts files in a directory."""
    assert discover_files._count_files(temp_directory_with_files) == 5


def test_iter_files(temp_directory_with_files):
    """Verify that the file iterator yields all files in a directory."""
    files = list(discover_files._iter_files(temp_directory_with_files))
    assert len(files) == 5
    expected_paths = {
        temp_directory_with_files / "archive.zip",
        temp_directory_with_files / "file1.txt",
        temp_directory_with_files / "file2.jpg",
        temp_directory_with_files / "file3.pdf",
        temp_directory_with_files / "subdir" / "file4.mov",
    }
    assert set(files) == expected_paths


@patch("src.discover_files.magic", None)
def test_create_mime_detector_magic_unavailable():
    """Verify that the MIME detector is None when python-magic is not installed."""
    assert discover_files._create_mime_detector() is None


@patch("src.discover_files.magic")
def test_create_mime_detector_magic_available(mock_magic):
    """Verify that a Magic object is created when python-magic is available."""
    mock_magic.Magic.return_value = "magic_instance"
    assert discover_files._create_mime_detector() == "magic_instance"


@patch("src.discover_files.magic")
def test_detect_mime_type_with_magic(mock_magic, temp_directory_with_files):
    """Verify MIME type detection using the 'magic' library."""
    mock_detector = MagicMock()
    mock_detector.from_file.return_value = "image/jpeg"
    mock_magic.Magic.return_value = mock_detector

    test_file = temp_directory_with_files / "file2.jpg"
    mime_type = discover_files._detect_mime_type(test_file, mock_detector)
    assert mime_type == "image/jpeg"
    mock_detector.from_file.assert_called_once_with(str(test_file))


@patch("src.discover_files.magic", None)
def test_detect_mime_type_fallback(temp_directory_with_files):
    """Verify MIME type detection falls back to the mimetypes library."""
    test_file = temp_directory_with_files / "file1.txt"
    mime_type = discover_files._detect_mime_type(test_file, None)
    assert mime_type == "text/plain"


def test_detect_mime_type_unknown(temp_directory_with_files):
    """Verify fallback to 'application/octet-stream' for unknown types."""
    # Create a file with an extension that mimetypes won't recognize
    unknown_file = temp_directory_with_files / "file.unknown"
    unknown_file.touch()

    # Ensure mimetypes.guess_type returns None
    with patch("mimetypes.guess_type", return_value=(None, None)):
        mime_type = discover_files._detect_mime_type(unknown_file, None)
        assert mime_type == "application/octet-stream"


def test_create_file_manifest_success(temp_directory_with_files):
    """Verify that a manifest is created correctly for a directory."""
    manifest_path = temp_directory_with_files / "manifest.json"
    records = discover_files.create_file_manifest(
        temp_directory_with_files, manifest_path
    )

    assert len(records) == 5
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest_data = json.load(f)
    assert len(manifest_data) == 5
    assert manifest_data[0]["file_name"] == "archive.zip"


def test_create_file_manifest_empty_directory(temp_directory_with_files):
    """Verify that an empty manifest is created for an empty directory."""
    empty_dir = temp_directory_with_files / "empty"
    empty_dir.mkdir()
    manifest_path = temp_directory_with_files / "manifest.json"

    records = discover_files.create_file_manifest(empty_dir, manifest_path)
    assert records == []


def test_create_file_manifest_non_existent_directory():
    """Verify that a ValueError is raised for a non-existent directory."""
    with pytest.raises(ValueError):
        discover_files.create_file_manifest(Path("/non/existent/dir"), Path("manifest.json"))


def test_create_file_manifest_max_files(temp_directory_with_files):
    """Verify that the max_files parameter correctly limits the number of files."""
    manifest_path = temp_directory_with_files / "manifest.json"
    records = discover_files.create_file_manifest(
        temp_directory_with_files, manifest_path, max_files=2
    )
    assert len(records) == 2


@patch("src.discover_files._calculate_sha256", side_effect=Exception("Test error"))
def test_create_file_manifest_error_handling(
    mock_sha256, temp_directory_with_files, caplog
):
    """Verify that file processing errors are logged and skipped."""
    manifest_path = temp_directory_with_files / "manifest.json"
    with caplog.at_level(logging.ERROR):
        records = discover_files.create_file_manifest(
            temp_directory_with_files, manifest_path
        )
        assert len(records) == 0  # No records should be created
        assert "Failed to process file" in caplog.text
        assert "Test error" in caplog.text


def test_build_parser():
    """Verify that the argument parser is configured correctly."""
    parser = discover_files._build_parser()
    args = parser.parse_args(["/some/dir"])
    assert args.root_directory == Path("/some/dir")
    assert args.manifest_path == Path("data/manifest.json")
    assert args.max_files == 0
    assert not args.verbose
    assert not args.debug


@patch("src.discover_files.create_file_manifest")
@patch("src.discover_files.configure_logging")
def test_main_success(
    mock_configure_logging, mock_create_manifest, temp_directory_with_files
):
    """Verify that the main function orchestrates file discovery correctly."""
    argv = ["--max-files", "10", str(temp_directory_with_files)]
    mock_create_manifest.return_value = [{}, {}]  # Simulate two records

    return_code = discover_files.main(argv)

    assert return_code == 0
    mock_configure_logging.assert_called_once()
    mock_create_manifest.assert_called_once()
    # Check that the CLI arguments are passed correctly
    call_args = mock_create_manifest.call_args[0]
    assert call_args[0] == temp_directory_with_files
    assert call_args[2] == 10


@patch("src.discover_files.create_file_manifest", side_effect=Exception("Test error"))
@patch("src.discover_files.configure_logging")
def test_main_failure(
    mock_configure_logging, mock_create_manifest, temp_directory_with_files, caplog
):
    """Verify that the main function handles exceptions and returns a non-zero exit code."""
    argv = [str(temp_directory_with_files)]
    # We are mocking configure_logging, so we need to add a handler to capture logs
    logger = logging.getLogger("src.discover_files")
    logger.addHandler(caplog.handler)

    with caplog.at_level(logging.ERROR):
        return_code = discover_files.main(argv)
        assert return_code == 1
        assert "Failed to create file manifest: Test error" in caplog.text