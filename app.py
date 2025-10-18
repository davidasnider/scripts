import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import cv2
import ollama
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

from src.filters import apply_manifest_filters
from src.logging_utils import configure_logging
from src.schema import AnalysisName

# Truncation length for summary/description fields in tables
SUMMARY_TRUNCATE_LENGTH = 120
ACCESS_TABLE_PREVIEW_ROWS = 50

configure_logging()
logger = logging.getLogger("file_catalog.app")
logger.info("Streamlit UI initialized")

# Configure Streamlit page
st.set_page_config(page_title="Local AI Digital Archive", layout="wide")

# Main title
st.title("🔎 Local AI Digital Archive")


# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL = config["models"]["embedding_model"]
LLM_MODEL = config["models"]["text_analyzer"]
logger.debug(
    "Loaded configuration (embedding_model=%s, llm_model=%s)",
    EMBEDDING_MODEL,
    LLM_MODEL,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I help you explore your files?",
            "sources": [],
        }
    ]

if "filters" not in st.session_state:
    st.session_state.filters = {
        "file_type": [],
        "hide_nsfw": True,
        "red_flags": False,
        "fully_analyzed": False,
        "analysis_tasks": [],
        "no_tasks_complete": False,
    }


_WIDGET_KEY_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z_]+")


def _build_widget_key(prefix: str, identifier: str) -> str:
    """Create a deterministic and Streamlit-friendly widget key."""

    safe_identifier = _WIDGET_KEY_SANITIZE_PATTERN.sub("_", identifier)
    return f"{prefix}_{safe_identifier}"


def create_thumbnail(file_path: str, mime_type: str, max_size: tuple = (200, 200)):
    """Create a thumbnail for display in the app."""
    try:
        logger.debug("Creating thumbnail for %s (%s)", file_path, mime_type)
        file_path = Path(file_path)
        if not file_path.exists():
            logger.debug("Skipping thumbnail; file not found: %s", file_path)
            return None

        if mime_type.startswith("image/"):
            # For images, create thumbnail directly
            image = Image.open(file_path)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image

        elif mime_type.startswith("video/"):
            # For videos, extract first frame and create thumbnail
            logger.debug("Extracting video thumbnail for %s", file_path)
            cap = cv2.VideoCapture(str(file_path))
            ret, frame = cap.read()
            cap.release()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                return image

    except Exception as e:
        st.error(f"Failed to create thumbnail for {file_path}: {e}")
        logger.exception("Failed to create thumbnail for %s", file_path)

    return None


def display_source_with_thumbnail(source: dict, index: int):
    """Display a source with thumbnail if it's an image or video."""
    metadata = source.get("metadata", {})
    file_path = source.get("file_path", metadata.get("file_path", "N/A"))
    mime_type = metadata.get("mime_type", "")

    # Create columns for thumbnail and text
    if mime_type.startswith(("image/", "video/")):
        col1, col2 = st.columns([1, 3])

        with col1:
            if mime_type.startswith("video/"):
                # For videos, try to show multiple frames if they exist
                frames_dir = Path("data/frames")
                frame_files = (
                    list(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []
                )

                if frame_files:
                    # Show first few frames as thumbnails
                    st.write("📹 **Video Frames:**")
                    frame_files_sorted = sorted(frame_files)[:3]  # Show max 3 frames

                    for frame_file in frame_files_sorted:
                        try:
                            frame_image = Image.open(frame_file)
                            frame_image.thumbnail((150, 150), Image.Resampling.LANCZOS)
                            st.image(
                                frame_image,
                                caption=f"Frame {frame_file.stem.split('_')[-1]}",
                            )
                        except Exception:
                            continue
                else:
                    # Fallback to extracting first frame
                    thumbnail = create_thumbnail(file_path, mime_type)
                    if thumbnail:
                        st.image(thumbnail, caption="📹 Video thumbnail")
                    else:
                        st.write("📹 Thumbnail unavailable")
            else:
                # For images, show thumbnail
                thumbnail = create_thumbnail(file_path, mime_type)
                if thumbnail:
                    st.image(thumbnail, caption="🖼️ Image")

                    # Add button to view full size
                    if st.button(
                        "View Full Size",
                        key=_build_widget_key("fullsize", f"{index}_{str(file_path)}"),
                    ):
                        try:
                            full_image = Image.open(file_path)
                            st.image(
                                full_image, caption=f"Full size: {Path(file_path).name}"
                            )
                        except Exception as e:
                            st.error(f"Could not load full image: {e}")
                            logger.exception(
                                "Could not load full image for %s", file_path
                            )
                else:
                    st.write("🖼️ Thumbnail unavailable")

        with col2:
            st.write(
                f"**File:** {Path(file_path).name if file_path != 'N/A' else 'N/A'}"
            )
            st.write(f"**Type:** {mime_type}")
            if "page_number" in source:
                st.write(f"**Page:** {source['page_number']}")
            st.write(f"**Snippet:** {source.get('content_snippet', 'N/A')[:300]}...")
    else:
        # Non-media files - display normally
        st.write(f"**File:** {Path(file_path).name if file_path != 'N/A' else 'N/A'}")
        st.write(f"**Type:** {mime_type}")
        if "page_number" in source:
            st.write(f"**Page:** {source['page_number']}")
        st.write(f"**Snippet:** {source.get('content_snippet', 'N/A')}")

    file_path_str = str(file_path)
    path_obj = Path(file_path_str)
    if file_path_str != "N/A":
        if path_obj.exists():
            if st.button(
                "Open file",
                key=_build_widget_key("open_source", f"{index}_{file_path_str}"),
            ):
                if open_file_with_system(file_path_str):
                    st.success("Opening file with the default application.")
        else:
            st.caption("File not found on disk; cannot open.")


def open_file_with_system(file_path: str) -> bool:
    """Launch a file with the system default handler."""
    path = Path(file_path)
    if not path.exists():
        st.error("File could not be located on disk.")
        logger.warning("Open file requested but missing: %s", file_path)
        return False

    try:
        if os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        logger.info("Opening file with system handler: %s", file_path)
        return True
    except Exception as err:
        st.error(f"Failed to open file: {err}")
        logger.exception("Failed to open file %s", file_path)
        return False


def format_bytes(size: int | float | None) -> str:
    """Convert bytes to a human readable string."""
    if size is None:
        return "Unknown"
    if not isinstance(size, (int, float)):
        raise TypeError(f"Expected int or float for size, got {type(size).__name__}")

    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(size)
    for unit in units:
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} EB"


def extract_selected_rows(table_state: Any) -> list[int]:
    """Normalize Streamlit table selection data into a list of row indices."""
    if table_state is None:
        return []

    if hasattr(table_state, "selection"):
        rows = table_state.selection.get("rows", [])
    elif isinstance(table_state, str):
        try:
            parsed_state = json.loads(table_state)
        except json.JSONDecodeError:
            return []
        rows = parsed_state.get("selection", {}).get("rows", [])
    elif isinstance(table_state, dict):
        rows = table_state.get("selection", {}).get("rows", [])
    else:
        return []

    if isinstance(rows, dict):
        rows = list(rows.keys())

    if isinstance(rows, (list, tuple, set)):
        return [int(idx) for idx in rows]

    # Handle other iterable views (e.g., dict_values) that are not list/tuple/set
    try:
        if rows is not None and not isinstance(rows, (str, bytes)):
            rows_iter = list(rows)  # type: ignore[arg-type]
            if rows_iter:
                return [int(idx) for idx in rows_iter]
    except TypeError:
        pass

    if rows is None:
        return []

    return [int(rows)]


def render_related_file_table(
    table_rows: list[dict[str, str]],
    files: list[dict[str, Any]],
    *,
    select_state_key: str,
    table_state_key: str,
    last_selection_state_key: str,
    log_label: str,
) -> None:
    """Render shared selectbox + dataframe controls for related file tables."""
    state_key = select_state_key
    widget_key = f"{select_state_key}_widget"

    if state_key not in st.session_state:
        st.session_state[state_key] = -1

    labels = [f"{row['File']} ({row['Directory']})" for row in table_rows]
    selection_options = [-1] + list(range(len(files)))
    try:
        select_default = selection_options.index(st.session_state[state_key])
    except ValueError:
        select_default = 0

    def format_option(idx: int) -> str:
        if idx == -1:
            return "-- Select a file --"
        if 0 <= idx < len(labels):
            return labels[idx]
        return f"File {idx}"

    selection = st.selectbox(
        "Open file details",
        options=selection_options,
        format_func=format_option,
        index=select_default,
        key=widget_key,
    )
    if selection != -1 and 0 <= selection < len(files):
        selected_path = files[selection].get("file_path")
        st.session_state.file_browser_selected_file = selected_path
        st.session_state[state_key] = selection
    else:
        st.session_state[state_key] = -1

    table_df = pd.DataFrame(table_rows)
    table_state = st.dataframe(
        table_df,
        hide_index=True,
        width="stretch",
        selection_mode="single-row",
        on_select="rerun",
        key=table_state_key,
    )

    widget_state = st.session_state.get(table_state_key)
    state_payload = widget_state if widget_state is not None else table_state

    selected_rows = extract_selected_rows(state_payload)
    previous_rows = st.session_state.get(last_selection_state_key, [])
    if selected_rows != previous_rows:
        st.session_state[last_selection_state_key] = selected_rows
        if selected_rows:
            selected_index = selected_rows[0]
            if 0 <= selected_index < len(files):
                selected_path = files[selected_index].get("file_path")
                st.session_state.file_browser_selected_file = selected_path
                st.session_state[state_key] = selected_index
                if selected_path:
                    logger.info("%s file selected: %s", log_label, selected_path)
        else:
            st.session_state[state_key] = -1
            st.session_state.file_browser_selected_file = None
            logger.info("%s selection cleared", log_label)


def make_table_rows(files: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Create a list of table rows from a list of file entries."""
    return [
        {
            "File": item.get("file_name") or Path(item.get("file_path", "")).name,
            "Directory": str(Path(item.get("file_path", "")).parent),
            "Summary": (item.get("summary") or item.get("description") or "")[
                :SUMMARY_TRUNCATE_LENGTH
            ],
        }
        for item in files
    ]


def get_entry_display_name(item: dict[str, Any]) -> str:
    """Get the display name for a file entry, used for sorting."""
    return (item.get("file_name") or Path(item.get("file_path", "")).name or "").lower()


def compute_directory_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build an index of directories, their child folders, and contained files."""
    directory_files: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    subdir_map: defaultdict[str, set[str]] = defaultdict(set)

    for entry in entries:
        file_path = entry.get("file_path")
        if not file_path:
            continue

        path_obj = Path(file_path)
        dir_path = str(path_obj.parent)
        directory_files[dir_path].append(entry)

        parents = list(path_obj.parents)
        for child, parent in zip(parents, parents[1:]):
            subdir_map[str(parent)].add(str(child))

    all_dirs: set[str] = set(directory_files.keys()) | set(subdir_map.keys())
    for subdirs in subdir_map.values():
        all_dirs.update(subdirs)

    directory_index: dict[str, dict[str, Any]] = {}
    for dir_path in all_dirs:
        dir_obj = Path(dir_path)
        parent_obj = dir_obj.parent
        parent_str = str(parent_obj) if parent_obj != dir_obj else None

        files = sorted(
            directory_files.get(dir_path, []),
            key=get_entry_display_name,
        )
        subdirs = sorted(
            subdir_map.get(dir_path, set()),
            key=lambda path: (Path(path).name or path).lower(),
        )

        directory_index[dir_path] = {
            "path": dir_path,
            "name": dir_obj.name or dir_path,
            "parent": parent_str,
            "subdirs": subdirs,
            "files": files,
            "file_count": len(files),
        }

    return directory_index


def compute_mime_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group files by MIME type for quick lookup."""
    mime_map: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        mime_type = entry.get("mime_type") or "unknown/unknown"
        mime_map[mime_type].append(entry)

    mime_index: dict[str, dict[str, Any]] = {}
    for mime_type, files in mime_map.items():
        mime_index[mime_type] = {
            "mime_type": mime_type,
            "count": len(files),
            "files": sorted(
                files,
                key=get_entry_display_name,
            ),
        }

    return dict(
        sorted(mime_index.items(), key=lambda item: (-item[1]["count"], item[0]))
    )


def compute_people_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group files by people mentioned in metadata."""
    people_map: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        people = entry.get("mentioned_people") or []
        for person in people:
            if not person:
                continue
            person_name = str(person).strip()
            if not person_name:
                continue
            people_map[person_name].append(entry)

    people_index: dict[str, dict[str, Any]] = {}
    for person_name, files in people_map.items():
        people_index[person_name] = {
            "person": person_name,
            "count": len(files),
            "files": sorted(
                files,
                key=get_entry_display_name,
            ),
        }

    return dict(
        sorted(
            people_index.items(),
            key=lambda item: (-item[1]["count"], item[0].lower()),
        )
    )


def compute_nsfw_index(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Create an index of NSFW files."""
    nsfw_files = [entry for entry in entries if entry.get("is_nsfw")]
    return {
        "count": len(nsfw_files),
        "files": sorted(
            nsfw_files,
            key=get_entry_display_name,
        ),
    }


@st.cache_data(show_spinner=False)
def load_manifest_assets(manifest_path: str = "data/manifest.json") -> dict[str, Any]:
    """Load manifest entries and build reusable indexes."""
    # manifest_path argument participates in the cache key to avoid stale cache
    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            entries: list[dict[str, Any]] = json.load(file)
    except FileNotFoundError:
        logger.warning("Manifest not found at %s", manifest_path)
        return {
            "entries": [],
            "path_lookup": {},
            "directory_index": {},
            "mime_index": {},
        }
    except json.JSONDecodeError:
        logger.exception("Manifest at %s is not valid JSON", manifest_path)
        return {
            "entries": [],
            "path_lookup": {},
            "directory_index": {},
            "mime_index": {},
        }

    directory_index = compute_directory_index(entries)
    mime_index = compute_mime_index(entries)
    path_lookup = {
        entry["file_path"]: entry for entry in entries if entry.get("file_path")
    }

    logger.info(
        "Manifest loaded (files=%d, directories=%d, mime_types=%d)",
        len(entries),
        len(directory_index),
        len(mime_index),
    )

    return {
        "entries": entries,
        "path_lookup": path_lookup,
        "directory_index": directory_index,
        "mime_index": mime_index,
    }


def render_file_detail(file_entry: dict[str, Any] | None, filtered_out: bool = False):
    """Show metadata, summary, and previews for a selected file."""
    st.markdown("### File Details")

    if not file_entry:
        _render_file_detail_empty(filtered_out)
        return

    if filtered_out:
        st.warning("This file is hidden by the current filters; showing full metadata.")

    file_path_str = file_entry.get("file_path") or "Unknown path"
    file_name = (
        file_entry.get("file_name") or Path(file_path_str).name or "Unknown name"
    )
    mime_type = file_entry.get("mime_type") or "unknown/unknown"

    _render_file_metadata(file_entry, file_path_str, file_name, mime_type)
    _render_file_summary_description(file_entry)
    preview_rendered = _render_file_previews(
        file_entry, file_path_str, file_name, mime_type
    )

    if not preview_rendered and mime_type.startswith("image/"):
        st.warning("Image preview unavailable; the file could not be located.")
    elif not preview_rendered and file_entry.get("extracted_frames"):
        st.warning("Extracted frame references were found but the images are missing.")

    if file_entry.get("analysis_tasks"):
        with st.expander("Analysis Tasks"):
            st.json(file_entry["analysis_tasks"])

    if file_entry.get("potential_red_flags"):
        with st.expander("Potential Red Flags"):
            st.write(file_entry["potential_red_flags"])


def _render_file_detail_empty(filtered_out: bool):
    if filtered_out:
        st.warning("The previously selected file is hidden by the current filters.")
    else:
        st.info("Select a file to see its details.")


def _render_file_metadata(
    file_entry: dict[str, Any], file_path_str: str, file_name: str, mime_type: str
):
    st.subheader(file_name)
    st.code(file_path_str)

    file_path = Path(file_path_str)
    file_exists = file_path.exists()
    if file_exists:
        if st.button(
            "Open in default application",
            key=_build_widget_key("open_detail", file_path_str),
        ):
            if open_file_with_system(file_path_str):
                st.success("Opening file in the system viewer.")
    else:
        st.caption("File not found on disk; opening is unavailable.")

    file_size = file_entry.get("file_size")
    last_modified = file_entry.get("last_modified")
    size_text = (
        format_bytes(file_size) if isinstance(file_size, (int, float)) else "Unknown"
    )
    meta_line = f"**Type:** {mime_type} · **Size:** {size_text}"
    if isinstance(last_modified, (int, float)):
        timestamp = datetime.fromtimestamp(last_modified)
        meta_line += f" · **Modified:** {timestamp:%Y-%m-%d %H:%M:%S}"
    st.markdown(meta_line)


def _render_file_summary_description(file_entry: dict[str, Any]):
    summary = file_entry.get("summary")
    description = file_entry.get("description")

    st.markdown("**Summary**")
    st.write(summary if summary else "_No summary available._")

    st.markdown("**Description**")
    st.write(description if description else "_No description available._")


def _render_file_previews(
    file_entry: dict[str, Any], file_path_str: str, file_name: str, mime_type: str
) -> bool:
    file_path = Path(file_path_str)
    preview_rendered = False

    if mime_type.startswith("image/") and file_path.exists():
        st.image(str(file_path), caption=file_name)
        preview_rendered = True

    frames = file_entry.get("extracted_frames") or []
    existing_frames = [
        Path(frame) for frame in frames if frame and Path(frame).exists()
    ]
    if existing_frames:
        st.markdown("**Extracted Frames**")
        columns = st.columns(min(3, len(existing_frames)))
        for idx, frame_path in enumerate(existing_frames[:9]):
            with columns[idx % len(columns)]:
                st.image(str(frame_path), caption=frame_path.name)
        preview_rendered = True

    return preview_rendered


def render_directory_browser(directory_index: dict[str, dict[str, Any]]):
    """Render controls for browsing by directory."""
    if not directory_index:
        st.info("No directories available for the current filters.")
        return

    if "file_browser_selected_dir" not in st.session_state:
        default_dir = max(
            directory_index.values(),
            key=lambda item: item["file_count"],
            default=None,
        )
        st.session_state.file_browser_selected_dir = (
            default_dir["path"] if default_dir else next(iter(directory_index))
        )

    selected_dir = st.session_state.get("file_browser_selected_dir")
    if selected_dir not in directory_index:
        selected_dir = next(iter(directory_index))
        st.session_state.file_browser_selected_dir = selected_dir

    directory_info = directory_index[selected_dir]

    parent_path = directory_info.get("parent")

    st.caption(f"Location: `{selected_dir}`")

    st.caption(f"{directory_info['file_count']} file(s) in this directory.")

    subdirs = directory_info.get("subdirs", [])
    files = directory_info.get("files", [])

    if not subdirs and not files:
        st.info("This directory does not contain any items matching the filters.")
        return

    st.markdown("**Folder Contents**")

    combined_entries: list[dict[str, Any]] = []
    table_rows: list[dict[str, str]] = []

    selected_file_path = st.session_state.get("file_browser_selected_file")

    if parent_path and parent_path in directory_index:
        parent_name = Path(parent_path).name or parent_path
        combined_entries.append(
            {"path": parent_path, "is_dir": True, "is_parent": True}
        )
        table_rows.append(
            {
                "Name": "↩︎ Parent Directory",
                "Details": parent_name,
                "Summary": "",
            }
        )

    for subdir in subdirs:
        child_info = directory_index.get(subdir, {})
        item_count = child_info.get("file_count", 0)
        display_name = Path(subdir).name or subdir
        combined_entries.append({"path": subdir, "is_dir": True})
        table_rows.append(
            {
                "Name": f"📁 {display_name}",
                "Details": f"{item_count} item{'s' if item_count != 1 else ''}",
                "Summary": "",
            }
        )

    for item in files:
        file_path = item.get("file_path", "")
        file_name = item.get("file_name") or Path(file_path).name or "Unknown file"
        summary_text = (item.get("summary") or item.get("description") or "") or ""
        summary_text = summary_text[:SUMMARY_TRUNCATE_LENGTH]
        file_size = item.get("file_size")
        details = format_bytes(file_size) if isinstance(file_size, (int, float)) else ""

        combined_entries.append({"path": file_path, "is_dir": False})
        display_label = f"📄 {file_name}"
        if file_path == selected_file_path:
            display_label = f"✅ {display_label}"

        table_rows.append(
            {
                "Name": display_label,
                "Details": details,
                "Summary": summary_text,
            }
        )

    entry_lookup = {entry["path"]: entry for entry in combined_entries}
    paths = [entry["path"] for entry in combined_entries]

    display_df = pd.DataFrame(table_rows)

    action_state_map = st.session_state.get("directory_action_flags")
    if not isinstance(action_state_map, dict):
        action_state_map = {}

    action_values = [bool(action_state_map.get(path, False)) for path in paths]
    display_df.insert(0, "Action", action_values)
    display_df["Path"] = paths

    editor_state = st.data_editor(
        display_df,
        hide_index=True,
        num_rows="fixed",
        column_order=["Action", "Name", "Details", "Summary", "Path"],
        column_config={
            "Action": st.column_config.CheckboxColumn(
                label="", default=False, width="small"
            ),
            "Name": st.column_config.Column(disabled=True),
            "Details": st.column_config.Column(disabled=True),
            "Summary": st.column_config.Column(disabled=True),
            "Path": st.column_config.TextColumn(disabled=True),
        },
        width="stretch",
        key="directory_browser_table",
    )

    if editor_state is not None and "Action" in editor_state and "Path" in editor_state:
        current_flags = dict(
            zip(
                editor_state["Path"],
                editor_state["Action"].astype(bool),
            )
        )
    else:
        current_flags = {path: False for path in paths}

    previous_flags = action_state_map if isinstance(action_state_map, dict) else {}

    trigger_path = None
    for path, is_checked in current_flags.items():
        if is_checked and not previous_flags.get(path, False):
            trigger_path = path
            break

    if trigger_path:
        entry_meta = entry_lookup.get(trigger_path)
        if entry_meta:
            entry_path = entry_meta["path"]
            if entry_meta.get("is_parent"):
                st.session_state.file_browser_selected_dir = entry_path
                st.session_state.file_browser_selected_file = None
                logger.info("Directory view parent opened via action: %s", entry_path)
            elif entry_meta["is_dir"]:
                st.session_state.file_browser_selected_dir = entry_path
                st.session_state.file_browser_selected_file = None
                logger.info("Directory view folder opened via action: %s", entry_path)
            else:
                st.session_state.file_browser_selected_file = entry_path
                logger.info("Directory view file selected via action: %s", entry_path)

        st.session_state.directory_action_flags = {path: False for path in paths}
        st.rerun()
    else:
        st.session_state.directory_action_flags = current_flags


def render_mime_browser(
    mime_index: dict[str, dict[str, Any]], all_files: list[dict[str, Any]]
):
    """
    Render controls for browsing files by MIME type.

    Parameters:
        mime_index (dict[str, dict[str, Any]]): Index mapping MIME types to file info.
        all_files (list[dict[str, Any]]): List of all files after filtering.

    Notes:
        Selecting "ALL" will show all filtered files regardless of MIME type.
    """
    if not mime_index and not all_files:
        st.info("No MIME types found for the current filters.")
        return

    if "file_browser_selected_mime" not in st.session_state:
        st.session_state.file_browser_selected_mime = "ALL"

    options = ["ALL"] + list(mime_index.keys())
    try:
        default_index = options.index(st.session_state.file_browser_selected_mime)
    except ValueError:
        default_index = 0

    selected_mime = st.selectbox(
        "Select a MIME type",
        options,
        index=default_index,
        key="mime_type_select",
    )
    st.session_state.file_browser_selected_mime = selected_mime

    if selected_mime == "ALL":
        files = sorted(all_files, key=get_entry_display_name)
        st.caption(f"{len(files)} total file(s).")
    else:
        mime_info = mime_index.get(selected_mime, {})
        files = sorted(mime_info.get("files", []), key=get_entry_display_name)
        st.caption(f"{len(files)} file(s) with MIME type `{selected_mime}`.")

    table_rows = make_table_rows(files)

    render_related_file_table(
        table_rows,
        files,
        select_state_key="mime_file_select",
        table_state_key="mime_file_table",
        last_selection_state_key="mime_file_table_last_selection",
        log_label="MIME view",
    )


def render_people_browser(people_index: dict[str, dict[str, Any]]):
    """Render controls for browsing by mentioned people."""
    if not people_index:
        st.info("No people metadata available for the current filters.")
        return

    if "file_browser_selected_person" not in st.session_state:
        st.session_state.file_browser_selected_person = next(iter(people_index))

    people_names = list(people_index.keys())
    try:
        default_index = people_names.index(
            st.session_state.file_browser_selected_person
        )
    except ValueError:
        default_index = 0

    selected_person = st.selectbox(
        "Select a person mentioned",
        people_names,
        index=default_index,
        key="people_select",
    )
    st.session_state.file_browser_selected_person = selected_person

    person_info = people_index[selected_person]
    files = person_info.get("files", [])

    st.caption(f"{len(files)} file(s) mention **{selected_person}**.")

    table_rows = make_table_rows(files)

    render_related_file_table(
        table_rows,
        files,
        select_state_key="people_file_select",
        table_state_key="people_file_table",
        last_selection_state_key="people_file_table_last_selection",
        log_label="People view",
    )


def render_nsfw_browser(nsfw_index: dict[str, Any]):
    """Render controls for browsing NSFW files."""
    if not nsfw_index or not nsfw_index.get("files"):
        st.info("No NSFW files found for the current filters.")
        return

    files = nsfw_index.get("files", [])
    st.caption(f"{len(files)} NSFW file(s) found.")

    table_rows = make_table_rows(files)

    render_related_file_table(
        table_rows,
        files,
        select_state_key="nsfw_file_select",
        table_state_key="nsfw_file_table",
        last_selection_state_key="nsfw_file_table_last_selection",
        log_label="NSFW view",
    )


def render_file_browser(filter_state: dict[str, Any]):
    """Top-level renderer for the file browser tab."""
    manifest_assets = load_manifest_assets()
    entries = manifest_assets["entries"]

    if not entries:
        st.info(
            "No manifest entries were found. Run `uv run python src/discover_files.py` "
            "to index your files."
        )
        return

    filtered_entries = apply_manifest_filters(entries, filter_state)

    if not filtered_entries:
        st.info("No files match the current filters. Adjust them in the filters panel.")
        return

    directory_index = compute_directory_index(filtered_entries)
    mime_index = compute_mime_index(filtered_entries)
    people_index = compute_people_index(filtered_entries)
    nsfw_index = compute_nsfw_index(filtered_entries)
    filtered_lookup = {
        entry["file_path"]: entry
        for entry in filtered_entries
        if entry.get("file_path")
    }
    full_lookup = manifest_assets["path_lookup"]

    total_files = len(filtered_entries)
    total_directories = len(directory_index)
    total_mime_types = len(mime_index)
    total_people = len(people_index)
    total_nsfw = nsfw_index.get("count", 0)

    met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
    met_col1.metric("Files", f"{total_files:,}")
    met_col2.metric("Directories", f"{total_directories:,}")
    met_col3.metric("MIME types", f"{total_mime_types:,}")
    met_col4.metric("People", f"{total_people:,}")
    met_col5.metric("NSFW", f"{total_nsfw:,}")

    if "file_browser_view_mode" not in st.session_state:
        st.session_state.file_browser_view_mode = "Directory"
    elif st.session_state.file_browser_view_mode not in {
        "Directory",
        "MIME type",
        "People",
        "NSFW",
    }:
        st.session_state.file_browser_view_mode = "Directory"

    view_mode = st.radio(
        "Browse files by",
        ["Directory", "MIME type", "People", "NSFW"],
        horizontal=True,
        key="file_browser_view_mode",
    )

    nav_col, detail_col = st.columns((1.2, 1.8))
    with nav_col:
        if view_mode == "Directory":
            render_directory_browser(directory_index)
        elif view_mode == "MIME type":
            render_mime_browser(mime_index, filtered_entries)
        elif view_mode == "People":
            render_people_browser(people_index)
        elif view_mode == "NSFW":
            render_nsfw_browser(nsfw_index)

    selected_path = st.session_state.get("file_browser_selected_file")
    selected_entry = filtered_lookup.get(selected_path)
    filtered_out = False
    if not selected_entry and selected_path:
        selected_entry = full_lookup.get(selected_path)
        filtered_out = bool(selected_entry)

    with detail_col:
        render_file_detail(selected_entry, filtered_out=filtered_out)


# Inline filters panel at the top of the page
with st.container():
    (
        filter_col1,
        filter_col2,
        filter_col3,
        filter_col4,
        filter_col5,
        filter_col6,
    ) = st.columns([1.5, 0.5, 0.5, 0.5, 1, 1.5])
    with filter_col1:
        st.session_state.filters["file_type"] = st.multiselect(
            "File type:",
            options=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "image/jpeg",
                "image/png",
                "video/mp4",
                "video/quicktime",
            ],
            default=st.session_state.filters["file_type"],
        )
    with filter_col2:
        st.session_state.filters["hide_nsfw"] = st.checkbox(
            "Hide NSFW", value=st.session_state.filters["hide_nsfw"]
        )
    with filter_col3:
        st.session_state.filters["red_flags"] = st.checkbox(
            "Red flags",
            value=st.session_state.filters["red_flags"],
        )
    with filter_col4:
        st.session_state.filters["fully_analyzed"] = st.checkbox(
            "Fully analyzed", value=st.session_state.filters["fully_analyzed"]
        )
    with filter_col5:
        st.session_state.filters["no_tasks_complete"] = st.checkbox(
            "No analysis tasks (file complete)",
            value=st.session_state.filters["no_tasks_complete"],
        )
    with filter_col6:
        st.session_state.filters["analysis_tasks"] = st.multiselect(
            "Completed analyses:",
            options=[task.value for task in AnalysisName],
            default=st.session_state.filters["analysis_tasks"],
        )

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path="./data/chromadb")
    collection = chroma_client.get_collection(name="digital_archive")
except Exception as e:
    st.error(f"Failed to connect to ChromaDB: {e}")
    st.stop()


def build_chroma_filter(filter_state: dict) -> dict:
    """Build a ChromaDB where filter from the filter state dictionary."""
    logger.debug("Building Chroma filter from state: %s", filter_state)
    filters = []

    # Use variables to avoid string interpolation issues
    eq_op = "$" + "eq"
    in_op = "$" + "in"
    and_op = "$" + "and"

    if filter_state.get("hide_nsfw"):
        filters.append({"is_nsfw": {eq_op: False}})

    if filter_state.get("red_flags"):
        filters.append({"has_financial_red_flags": {eq_op: True}})

    if filter_state.get("file_type"):
        if len(filter_state["file_type"]) == 1:
            filters.append({"mime_type": {eq_op: filter_state["file_type"][0]}})
        else:
            filters.append({"mime_type": {in_op: filter_state["file_type"]}})

    if len(filters) == 1:
        result = filters[0]
    elif len(filters) > 1:
        result = {and_op: filters}
    else:
        result = {}

    logger.debug("Generated Chroma filter: %s", result)
    return result


def get_database_stats(filters: dict = None) -> dict:
    """Get statistics about the database contents."""
    try:
        logger.info("Fetching database stats (filters_applied=%s)", bool(filters))
        if filters:
            # Get filtered results for counting
            results = collection.get(where=filters, include=["metadatas"])
        else:
            # Get all results for counting
            results = collection.get(include=["metadatas"])

        total_count = len(results["metadatas"])

        # Load manifest to get proper summaries
        try:
            with open("data/manifest.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_lookup = {item["file_path"]: item for item in manifest}
        except (FileNotFoundError, json.JSONDecodeError):
            manifest_lookup = {}

        # Group by file type
        file_types = {}
        file_list = []

        for metadata in results["metadatas"]:
            mime_type = metadata.get("mime_type", "unknown")
            file_name = metadata.get("file_name", "Unknown")
            file_path = metadata.get("file_path", "Unknown")

            # Get summary from manifest
            manifest_data = manifest_lookup.get(file_path, {})
            summary = manifest_data.get("summary", "No summary available")
            if summary == "No summary available" or not summary.strip():
                # Fallback to description if summary is empty
                description = manifest_data.get(
                    "description", "No description available"
                )
                if description != "No description available" and description.strip():
                    summary = description

            # Count by type
            if mime_type in file_types:
                file_types[mime_type] += 1
            else:
                file_types[mime_type] = 1

            # Add to file list
            file_list.append(
                {
                    "name": file_name,
                    "path": file_path,
                    "type": mime_type,
                    "is_nsfw": metadata.get("is_nsfw", False),
                    "has_red_flags": metadata.get("has_financial_red_flags", False),
                    "summary": summary,
                }
            )

        stats = {
            "total_count": total_count,
            "file_types": file_types,
            "file_list": file_list,
        }
        logger.info(
            "Database stats ready (total=%d, distinct_types=%d)",
            total_count,
            len(file_types),
        )
        return stats
    except Exception as e:
        st.error(f"Failed to get database stats: {e}")
        logger.exception("Failed to compute database stats")
        return {"total_count": 0, "file_types": {}, "file_list": []}


def query_knowledge_base(query_text: str, filters: dict) -> list[dict]:
    """Query the ChromaDB collection for relevant documents."""
    try:
        preview = query_text[:80] + ("…" if len(query_text) > 80 else "")
        logger.info("Querying knowledge base (filters_applied=%s)", bool(filters))
        logger.debug("Query preview: %s", preview)
        # Generate embedding for the query
        query_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query_text)[
            "embedding"
        ]

        # Query the collection - limit to 3 most relevant results to reduce context
        if filters:
            results = collection.query(
                query_embeddings=[query_embedding],
                where=filters,
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

        # Format results
        sources = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            sources.append(
                {
                    "content_snippet": doc,
                    "file_path": metadata.get("file_path", "Unknown file"),
                    "metadata": metadata,
                    "relevance_score": 1
                    - distance,  # Convert distance to similarity score
                }
            )

        logger.info("Knowledge base returned %d source(s)", len(sources))
        return sources
    except Exception as e:
        st.error(f"Failed to query knowledge base: {e}")
        logger.exception("Knowledge base query failed")
        return []


def generate_response(
    query_text: str, context: list[dict], chat_history: list[dict]
) -> iter:
    """Generate a streaming response from the LLM."""
    try:
        logger.info(
            "Generating response (context_sources=%d, history_entries=%d)",
            len(context),
            len(chat_history),
        )
        # Limit and summarize context to avoid prompt truncation
        max_context_chars = 2000  # Keep context under 2000 chars
        summarized_context = []
        total_chars = 0

        for src in context:
            snippet = src["content_snippet"]
            file_name = src.get("file_path", "Unknown file")
            if file_name != "Unknown file":
                file_name = file_name.split("/")[-1]  # Just filename

            # Truncate very long snippets
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."

            context_entry = f"- {file_name}: {snippet}"

            if total_chars + len(context_entry) > max_context_chars:
                if not summarized_context:  # Include at least one result
                    summarized_context.append(context_entry)
                break

            summarized_context.append(context_entry)
            total_chars += len(context_entry)

        context_str = "\n".join(summarized_context)
        logger.debug(
            "Prepared context (characters=%d, segments=%d)",
            len(context_str),
            len(summarized_context),
        )

        # Limit chat history as well
        history_str = "\n".join(
            [
                (
                    f"{msg['role']}: {msg['content'][:200]}..."
                    if len(msg["content"]) > 200
                    else f"{msg['role']}: {msg['content']}"
                )
                for msg in chat_history[-3:]
            ]  # Last 3 messages, truncated
        )

        prompt = (
            "You are a helpful assistant for exploring digital archives. "
            "Answer based on the provided file context.\n\n"
        )

        if history_str:
            prompt += "Recent Conversation:\n" + history_str + "\n\n"

        prompt += (
            "Relevant Files:\n" + context_str + "\n\n"
            "User Question: " + query_text + "\n\n"
            "Provide a helpful response based on the file content above. "
            "Mention specific filenames when referring to content."
        )

        # Debug: Check prompt length
        prompt_length = len(prompt)
        if prompt_length > 3000:
            st.warning(
                f"Large prompt ({prompt_length} chars); response may be truncated."
            )
        logger.debug("Prompt length for response: %d", prompt_length)

        # Call Ollama with streaming
        response = ollama.chat(
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True
        )

        logger.info("LLM streaming response started")
        return response
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        logger.exception("Failed to generate LLM response")
        return iter([])  # Return empty iterator


chat_tab, browser_tab = st.tabs(["Chat", "File Browser"])

with chat_tab:
    # Display existing conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources") and len(message["sources"]) > 0:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        display_source_with_thumbnail(source, i)
                        if i < len(message["sources"]) - 1:
                            st.write("---")
    prompt = st.chat_input("Ask about your archive")

    if prompt:
        prompt_preview = prompt[:SUMMARY_TRUNCATE_LENGTH] + (
            "…" if len(prompt) > SUMMARY_TRUNCATE_LENGTH else ""
        )
        logger.info("Received user prompt: %s", prompt_preview)

        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stats_keywords = [
                "how many",
                "total files",
                "count",
                "database stats",
                "how much",
                "all files",
                "files in",
                "archive",
                "database",
                "list files",
                "show files",
                "what files",
                "tell me about my files",
                "about my files",
                "my files",
            ]
            is_stats_query = any(
                keyword in prompt.lower() for keyword in stats_keywords
            )
            logger.debug("Stats query detection: %s", is_stats_query)

            if is_stats_query:
                logger.info("Handling statistics query")
                with st.spinner("Counting files in your archive..."):
                    chroma_filters = build_chroma_filter(st.session_state.filters)
                    stats = get_database_stats(
                        chroma_filters if chroma_filters else None
                    )

                response_text = "**Database Statistics:**\n\n"
                response_text += f"📁 **Total files:** {stats['total_count']}\n\n"

                if stats["file_types"]:
                    response_text += "**File types breakdown:**\n"
                    for file_type, count in sorted(stats["file_types"].items()):
                        plural = "s" if count != 1 else ""
                        response_text += f"- {file_type}: {count} file{plural}\n"
                    response_text += "\n"

                if stats["file_list"]:
                    response_text += "**Files in your archive:**\n\n"
                    for i, file_info in enumerate(stats["file_list"], 1):
                        name = file_info["name"]
                        file_path = file_info["path"]
                        file_type = file_info["type"]
                        summary = file_info.get("summary", "No summary available")
                        flags = []
                        if file_info["is_nsfw"]:
                            flags.append("NSFW")
                        if file_info["has_red_flags"]:
                            flags.append("🚩 Financial flags")

                        flag_text = f" ({', '.join(flags)})" if flags else ""

                        response_text += f"{i}. **{name}** - {file_type}{flag_text}\n"
                        response_text += f"   📁 `{file_path}`\n"
                        response_text += f"   *{summary}*\n\n"

                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text, "sources": []}
                )

            else:
                logger.info("Handling knowledge base query")
                with st.spinner("Searching your archive..."):
                    chroma_filters = build_chroma_filter(st.session_state.filters)
                    context = query_knowledge_base(prompt, chroma_filters)

                if not context:
                    logger.info("No matching context found for prompt")
                    st.warning(
                        "No relevant information found in your archive for this query."
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "No relevant information found in your archive.",
                            "sources": [],
                        }
                    )
                else:
                    response_placeholder = st.empty()
                    full_response = ""

                    response_stream = generate_response(
                        prompt, context, st.session_state.messages[:-1]
                    )
                    for chunk in response_stream:
                        if chunk["message"]["content"]:
                            full_response += chunk["message"]["content"]
                            response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)
                    logger.info(
                        "Streaming response completed (length=%d)", len(full_response)
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "sources": context,
                        }
                    )

        st.rerun()

with browser_tab:
    render_file_browser(st.session_state.filters)
