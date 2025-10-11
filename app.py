from pathlib import Path

import chromadb
import cv2
import ollama
import streamlit as st
import yaml
from PIL import Image

# Configure Streamlit page
st.set_page_config(page_title="Local AI Digital Archive", layout="wide")

# Main title
st.title("🔎 Local AI Digital Archive")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL = config["models"]["embedding_model"]
LLM_MODEL = config["models"]["text_analyzer"]

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
    st.session_state.filters = {"file_type": [], "hide_nsfw": True, "red_flags": False}


def create_thumbnail(file_path: str, mime_type: str, max_size: tuple = (200, 200)):
    """Create a thumbnail for display in the app."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        if mime_type.startswith("image/"):
            # For images, create thumbnail directly
            image = Image.open(file_path)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image

        elif mime_type.startswith("video/"):
            # For videos, extract first frame and create thumbnail
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
                        "View Full Size", key=f"fullsize_{index}_{hash(file_path)}"
                    ):
                        try:
                            full_image = Image.open(file_path)
                            st.image(
                                full_image, caption=f"Full size: {Path(file_path).name}"
                            )
                        except Exception as e:
                            st.error(f"Could not load full image: {e}")
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


# Sidebar for search filters
with st.sidebar:
    st.header("Search Filters")

    # File type filter
    st.session_state.filters["file_type"] = st.multiselect(
        "Filter by file type:",
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

    # Hide NSFW content checkbox
    st.session_state.filters["hide_nsfw"] = st.checkbox(
        "Hide NSFW content", value=st.session_state.filters["hide_nsfw"]
    )

    # Show only files with financial red flags checkbox
    st.session_state.filters["red_flags"] = st.checkbox(
        "Show only files with financial red flags",
        value=st.session_state.filters["red_flags"],
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources") and len(message["sources"]) > 0:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    display_source_with_thumbnail(source, i)
                    if i < len(message["sources"]) - 1:
                        st.write("---")

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path="./data/chromadb")
    collection = chroma_client.get_collection(name="digital_archive")
except Exception as e:
    st.error(f"Failed to connect to ChromaDB: {e}")
    st.stop()


def build_chroma_filter(filter_state: dict) -> dict:
    """Build a ChromaDB where filter from the filter state dictionary."""
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
        return filters[0]
    elif len(filters) > 1:
        return {and_op: filters}
    return {}


def get_database_stats(filters: dict = None) -> dict:
    """Get statistics about the database contents."""
    try:
        if filters:
            # Get filtered results for counting
            results = collection.get(where=filters, include=["metadatas"])
        else:
            # Get all results for counting
            results = collection.get(include=["metadatas"])

        total_count = len(results["metadatas"])

        # Load manifest to get proper summaries
        import json

        try:
            with open("data/manifest.json", "r") as f:
                manifest = json.load(f)
            # Create a lookup dict by file path
            manifest_lookup = {item["file_path"]: item for item in manifest}
        except Exception:
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

        return {
            "total_count": total_count,
            "file_types": file_types,
            "file_list": file_list,
        }
    except Exception as e:
        st.error(f"Failed to get database stats: {e}")
        return {"total_count": 0, "file_types": {}, "file_list": []}


def query_knowledge_base(query_text: str, filters: dict) -> list[dict]:
    """Query the ChromaDB collection for relevant documents."""
    try:
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

        return sources
    except Exception as e:
        st.error(f"Failed to query knowledge base: {e}")
        return []


def generate_response(
    query_text: str, context: list[dict], chat_history: list[dict]
) -> iter:
    """Generate a streaming response from the LLM."""
    try:
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

        # Call Ollama with streaming
        response = ollama.chat(
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True
        )

        return response
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return iter([])  # Return empty iterator


# Chat input handling
if prompt := st.chat_input("Ask me about your files..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        # Check if this is a database statistics query
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
        is_stats_query = any(keyword in prompt.lower() for keyword in stats_keywords)

        # Debug: show what we're checking
        st.write(f"Debug - Query: '{prompt.lower()}', Is stats query: {is_stats_query}")

        if is_stats_query:
            with st.spinner("Counting files in your archive..."):
                chroma_filters = build_chroma_filter(st.session_state.filters)
                stats = get_database_stats(chroma_filters if chroma_filters else None)

                # Create a comprehensive response
                response_text = "**Database Statistics:**\n\n"
                response_text += f"📁 **Total files:** {stats['total_count']}\n\n"

                if stats["file_types"]:
                    response_text += "**File types breakdown:**\n"
                    for file_type, count in sorted(stats["file_types"].items()):
                        response_text += (
                            f"- {file_type}: {count} file{'s' if count != 1 else ''}\n"
                        )
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

                        # Make it clean & copyable (Streamlit blocks file:// links)
                        response_text += f"{i}. **{name}** - {file_type}{flag_text}\n"
                        response_text += f"   📁 `{file_path}`\n"
                        response_text += f"   *{summary}*\n\n"

                st.markdown(response_text)

                # Add assistant message to session state
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text, "sources": []}
                )
        else:
            with st.spinner("Searching your archive..."):
                # Build filters and query knowledge base
                chroma_filters = build_chroma_filter(st.session_state.filters)
                context = query_knowledge_base(prompt, chroma_filters)

            if not context:
                st.warning(
                    "No relevant information found in your archive for this query."
                )
                # Add empty assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "No relevant information found in your archive.",
                        "sources": [],
                    }
                )
            else:
                # Create placeholder for streaming response
                response_placeholder = st.empty()
                full_response = ""

                # Stream the response
                response_stream = generate_response(
                    prompt, context, st.session_state.messages[:-1]
                )
                for chunk in response_stream:
                    if chunk["message"]["content"]:
                        full_response += chunk["message"]["content"]
                        response_placeholder.markdown(full_response + "▌")

                # Final display without cursor
                response_placeholder.markdown(full_response)

                # Add assistant message with sources to session state
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response, "sources": context}
                )

    # Rerun to update the UI with the new message
    st.rerun()
