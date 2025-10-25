from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.backup_manager import BackupManager

# Constants
MANIFEST_CONTENT = {"files": [{"path": "/test/file1"}, {"path": "/test/file2"}]}


@pytest.fixture
def backup_manager_fixture(tmp_path: Path) -> tuple[BackupManager, Path, Path]:
    """Fixture for setting up the BackupManager and test paths."""
    manifest_path = tmp_path / "manifest.json"
    backup_dir = tmp_path / "manifest_backups"
    backup_dir.mkdir()

    with manifest_path.open("w") as f:
        json.dump(MANIFEST_CONTENT, f)

    manager = BackupManager(
        manifest_path=manifest_path,
        backup_dir=backup_dir,
        interval_hours=1,
    )

    return manager, manifest_path, backup_dir


def create_backup(
    backup_dir: Path, timestamp: datetime, content: dict | None = None
) -> Path:
    """Helper function to create a backup file with a specific timestamp."""
    backup_path = (
        backup_dir / f"manifest-{timestamp.strftime('%Y%m%d-%H%M%S')}.json.bak"
    )

    if content is None:
        content = MANIFEST_CONTENT

    with backup_path.open("w") as f:
        if content:
            json.dump(content, f)

    # Set the modification time to the timestamp
    mod_time = time.mktime(timestamp.timetuple())
    backup_path.touch()
    import os

    os.utime(backup_path, (mod_time, mod_time))

    return backup_path


def test_backup_creation(backup_manager_fixture: tuple[BackupManager, Path, Path]):
    """Test that a backup is created successfully."""
    manager, _, backup_dir = backup_manager_fixture
    manager._create_backup()

    backups = list(backup_dir.glob("*.bak"))
    assert len(backups) == 1
    with backups[0].open("r") as f:
        assert json.load(f) == MANIFEST_CONTENT


def test_backup_validation(backup_manager_fixture: tuple[BackupManager, Path, Path]):
    """Test that invalid backups are correctly identified and removed."""
    manager, _, backup_dir = backup_manager_fixture

    # Create valid, zero-byte, and corrupted backups
    now = datetime.now()
    create_backup(backup_dir, now - timedelta(hours=1))
    zero_byte_backup = create_backup(backup_dir, now - timedelta(hours=2), content={})
    zero_byte_backup.write_text("")
    corrupted_backup = create_backup(
        backup_dir, now - timedelta(hours=3), content={}
    )
    corrupted_backup.write_text("{not valid json}")

    valid_backups = manager._get_valid_backups()
    assert len(valid_backups) == 1
    assert zero_byte_backup.exists() is False
    assert corrupted_backup.exists() is False


def test_tiered_pruning(backup_manager_fixture: tuple[BackupManager, Path, Path]):
    """Test the tiered backup retention algorithm."""
    manager, _, backup_dir = backup_manager_fixture

    now = datetime.now()
    timestamps = [
        now - timedelta(hours=1),
        now - timedelta(hours=2),
        now - timedelta(hours=3),  # Keep 3 most recent
        now - timedelta(days=1),  # Keep
        now - timedelta(days=1, hours=1),
        now - timedelta(days=2),  # Keep
        now - timedelta(days=2, hours=1),
        now - timedelta(days=8),  # Keep (weekly)
        now - timedelta(days=9),
        now - timedelta(days=15),  # Keep (weekly)
        now - timedelta(days=16),
        now - timedelta(days=31),  # Keep (monthly)
        now - timedelta(days=32),
        now - timedelta(days=60),  # Keep (monthly)
        now - timedelta(days=61),
        now - timedelta(days=90),  # Keep (monthly)
        now - timedelta(days=91),
        now - timedelta(days=120),  # Keep (monthly)
    ]

    for ts in timestamps:
        create_backup(backup_dir, ts)

    manager._prune_backups()

    remaining_backups = sorted(
        list(backup_dir.glob("*.bak")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    assert len(remaining_backups) <= manager.max_backups
    assert (
        len(remaining_backups) == 11
    )  # 3 recent + 2 daily + 2 weekly + 4 monthly
