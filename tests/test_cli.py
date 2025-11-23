from typer.testing import CliRunner

from src.file_catalog.__main__ import app

runner = CliRunner()


def test_scan_command():
    """Test the scan command."""
    result = runner.invoke(app, ["scan", "/tmp"])
    assert result.exit_code == 0
    assert "Scanning /tmp" in result.stdout


def test_analyze_command():
    """Test the analyze command."""
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 0
    assert "Analyzing catalog data" in result.stdout
