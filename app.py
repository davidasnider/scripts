import streamlit as st

# Configure Streamlit page
st.set_page_config(page_title="Local AI Digital Archive", layout="wide")

# Main title
st.title("🔎 Local AI Digital Archive")

# Constants
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:70b-instruct"

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

# Sidebar for search filters
with st.sidebar:
    st.header("Search Filters")

    # File type filter
    st.session_state.filters["file_type"] = st.multiselect(
        "Filter by file type:",
        options=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "image/jpeg",
            "image/png",
            "video/mp4",
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
                for source in message["sources"]:
                    st.write(f"**File:** {source.get('file_path', 'N/A')}")
                    if "page_number" in source:
                        st.write(f"**Page:** {source['page_number']}")
                    st.write(f"**Snippet:** {source.get('content_snippet', 'N/A')}")
                    st.write("---")
