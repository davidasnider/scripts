import json
from pathlib import Path

from typer.testing import CliRunner


def test_query_mime_type_filter(tmp_path: Path):
    """Verify that the --mime-type filter works correctly."""
    from scripts.query_manifest import app

    runner = CliRunner()

    manifest_data = [
        {
            "file_path": "/test/file1.txt",
            "file_name": "file1.txt",
            "mime_type": "text/plain",
            "file_size": 1,
            "last_modified": 1,
            "sha256": "a",
            "status": "complete",
        },
        {
            "file_path": "/test/file2.jpg",
            "file_name": "file2.jpg",
            "mime_type": "image/jpeg",
            "file_size": 1,
            "last_modified": 1,
            "sha256": "b",
            "status": "complete",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest_data, f)

    result = runner.invoke(
        app, ["--manifest-path", str(manifest_path), "--mime-type", "image/jpeg"]
    )
    assert result.exit_code == 0
    output_data = json.loads(result.stdout)
    assert len(output_data) == 1
    assert output_data[0]["file_name"] == "file2.jpg"
