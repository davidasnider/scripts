import streamlit as st

# Configure Streamlit page
st.set_page_config(page_title="Local AI Digital Archive", layout="wide")

# Main title
st.title("🔎 Local AI Digital Archive")

# Constants
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:70b-instruct"
