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
