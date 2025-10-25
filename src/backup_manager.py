from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copy2

# Constants
MANIFEST_PATH = Path("data/manifest.json")
MANIFEST_BACKUP_DIR = MANIFEST_PATH.parent / "manifest_backups"
BACKUP_INTERVAL_HOURS = 3
MAX_BACKUPS = 15

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages periodic backups of the manifest file."""

    def __init__(
        self,
        manifest_path: Path = MANIFEST_PATH,
        backup_dir: Path = MANIFEST_BACKUP_DIR,
        interval_hours: int = BACKUP_INTERVAL_HOURS,
        write_lock: threading.Lock | None = None,
    ):
        """Initializes the BackupManager."""
        self.manifest_path = manifest_path
        self.backup_dir = backup_dir
        self.interval_seconds = interval_hours * 3600
        self.max_backups = MAX_BACKUPS
        self.write_lock = write_lock
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Starts the background backup thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Backup thread is already running.")
            return

        logger.info("Starting manifest backup thread.")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the background backup thread."""
        if self._thread is None or not self._thread.is_alive():
            logger.info("Backup thread is not running.")
            return

        logger.info("Stopping manifest backup thread.")
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        """The main loop for the backup thread."""
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.backup_and_prune()
            except Exception as e:
                logger.error(f"Error in backup thread: {e}")

    def backup_and_prune(self) -> None:
        """Creates a new backup and prunes old ones."""
        logger.info("Running manifest backup and pruning.")
        self._create_backup()
        self._prune_backups()

    def _create_backup(self) -> None:
        """Creates a new timestamped backup of the manifest."""
        if not self.manifest_path.exists():
            return

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = self.backup_dir / f"manifest-{timestamp}.json.bak"

        try:
            if self.write_lock:
                with self.write_lock:
                    copy2(self.manifest_path, backup_path)
            else:
                copy2(self.manifest_path, backup_path)
            logger.info(f"Created manifest backup at {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create manifest backup: {e}")

    def _prune_backups(self) -> None:
        """Prunes old backups based on a tiered retention policy."""
        all_backups = self._get_valid_backups()
        if len(all_backups) <= self.max_backups:
            logger.debug(
                "Backup count (%d) is within the limit (%d), skipping prune.",
                len(all_backups),
                self.max_backups,
            )
            return

        to_keep: list[Path] = []
        now = datetime.now()

        # Tier 1: Keep the 3 most recent backups
        to_keep.extend(all_backups[:3])

        backups_to_consider = all_backups[3:]

        daily_kept = {datetime.fromtimestamp(p.stat().st_mtime).date() for p in to_keep}
        weekly_kept = {
            datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%W") for p in to_keep
        }
        monthly_kept = {
            datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m") for p in to_keep
        }

        for backup in backups_to_consider:
            backup_time = datetime.fromtimestamp(backup.stat().st_mtime)
            age = now - backup_time

            # Tier 2: Keep one per day for the last 7 days
            if age <= timedelta(days=7):
                day = backup_time.date()
                if day not in daily_kept:
                    to_keep.append(backup)
                    daily_kept.add(day)
                    weekly_kept.add(backup_time.strftime("%Y-%W"))
                    monthly_kept.add(backup_time.strftime("%Y-%m"))
                continue

            # Tier 3: Keep one per week for the last 30 days
            if age <= timedelta(days=30):
                week = backup_time.strftime("%Y-%W")
                if week not in weekly_kept:
                    to_keep.append(backup)
                    weekly_kept.add(week)
                    monthly_kept.add(backup_time.strftime("%Y-%m"))
                continue

            # Tier 4: Keep one per month
            month = backup_time.strftime("%Y-%m")
            if month not in monthly_kept:
                to_keep.append(backup)
                monthly_kept.add(month)

        # Final check against MAX_BACKUPS. If we still have too many, remove the oldest.
        if len(to_keep) > self.max_backups:
            to_keep = sorted(to_keep, key=lambda p: p.stat().st_mtime, reverse=True)
            to_keep = to_keep[: self.max_backups]

        backups_to_remove = set(all_backups) - set(to_keep)

        if backups_to_remove:
            logger.info(
                "Pruning %d backups to meet retention policy. Keeping %d.",
                len(backups_to_remove),
                len(to_keep),
            )

            for backup in backups_to_remove:
                try:
                    backup.unlink()
                    logger.debug(f"Removed old manifest backup {backup}")
                except Exception as e:
                    logger.error(f"Failed to remove backup {backup}: {e}")

    def _get_valid_backups(self) -> list[Path]:
        """Returns a list of valid, sorted backup files."""
        if not self.backup_dir.exists():
            return []

        backups = []
        for backup in self.backup_dir.glob("manifest-*.json.bak"):
            if self._is_valid_backup(backup):
                backups.append(backup)

        return sorted(backups, key=lambda p: p.stat().st_mtime, reverse=True)

    def _is_valid_backup(self, backup_path: Path) -> bool:
        """Checks if a backup file is valid."""
        if backup_path.stat().st_size == 0:
            logger.warning(f"Removing zero-byte backup: {backup_path}")
            backup_path.unlink()
            return False

        try:
            with backup_path.open("r", encoding="utf-8") as f:
                json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Removing corrupted backup {backup_path}: {e}")
            backup_path.unlink()
            return False

        return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    backup_manager = BackupManager(interval_hours=0.001)  # backup every ~3.6s
    backup_manager.start()

    try:
        time.sleep(15)
    finally:
        backup_manager.stop()
