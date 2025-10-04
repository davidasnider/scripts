from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import magic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled gracefully at runtime
    magic = None  # type: ignore[assignment]


STATUS_PENDING_EXTRACTION = "pending_extraction"


@dataclass(frozen=True)
class FileRecord:
    file_path: str
    file_name: str
    mime_type: str
    size_bytes: int
    status: str = STATUS_PENDING_EXTRACTION

    @classmethod
    def from_path(cls, path: Path, mime_detector: Any) -> "FileRecord":
        mime_type = _detect_mime_type(path, mime_detector)

        stat_info = path.stat()
        return cls(
            file_path=str(path),
            file_name=path.name,
            mime_type=mime_type,
            size_bytes=stat_info.st_size,
        )


def _iter_files(root_directory: Path) -> Iterable[Path]:
    for path in sorted(root_directory.rglob("*")):
        if path.is_file():
            yield path


def _create_mime_detector() -> Any:
    if magic is None:
        return None

    try:
        return magic.Magic(mime=True)
    except Exception as exc:  # pragma: no cover - depends on system libmagic
        print(
            f"Warning: python-magic could not initialize libmagic ({exc}). "
            "Falling back to mimetypes-based detection.",
            file=sys.stderr,
        )
        return None


def _detect_mime_type(path: Path, mime_detector: Any) -> str:
    if mime_detector is not None:
        try:
            return mime_detector.from_file(str(path))
        except Exception:
            pass

    guessed_type, _ = mimetypes.guess_type(str(path))
    if guessed_type:
        return guessed_type
    return "application/octet-stream"


def create_file_manifest(
    root_directory: Path, manifest_path: Path
) -> list[dict[str, object]]:
    root_directory = root_directory.expanduser().resolve()
    if not root_directory.is_dir():
        raise ValueError(
            f"Root directory does not exist or is not a directory: {root_directory}"
        )

    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    mime_detector = _create_mime_detector()
    try:
        records = [
            FileRecord.from_path(path, mime_detector)
            for path in _iter_files(root_directory)
        ]
    finally:
        if mime_detector is not None and hasattr(mime_detector, "close"):
            try:
                mime_detector.close()
            except Exception:
                pass

    manifest_data = [record.__dict__ for record in records]

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest_data, manifest_file, indent=2)
        manifest_file.write("\n")

    return manifest_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discover-files",
        description=(
            "Create a manifest describing files rooted " "at the specified directory."
        ),
    )
    parser.add_argument(
        "root_directory",
        type=Path,
        help="Directory to recursively scan for files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifest.json"),
        help=(
            "Location to write the generated manifest JSON "
            "(default: data/manifest.json)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    manifest = create_file_manifest(args.root_directory, args.manifest_path)
    print(f"Wrote {len(manifest)} entries to {args.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
