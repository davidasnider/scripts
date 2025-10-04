import chromadb
import ollama
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

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="digital_archive")
except Exception as e:
    st.error(f"Failed to connect to ChromaDB: {e}")
    st.stop()


def build_chroma_filter(filter_state: dict) -> dict:
    """Build a ChromaDB where filter from the filter state dictionary."""
    filters = []

    if filter_state.get("hide_nsfw"):
        filters.append({"nsfw_detected": {"$eq": False}})

    if filter_state.get("red_flags"):
        filters.append({"has_red_flags": {"$eq": True}})

    if filter_state.get("file_type"):
        if len(filter_state["file_type"]) == 1:
            filters.append({"mime_type": {"$eq": filter_state["file_type"][0]}})
        else:
            filters.append({"mime_type": {"$in": filter_state["file_type"]}})

    if filters:
        return {"$and": filters}
    return {}


def query_knowledge_base(query_text: str, filters: dict) -> list[dict]:
    """Query the ChromaDB collection for relevant documents."""
    try:
        # Generate embedding for the query
        query_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query_text)[
            "embedding"
        ]

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            where=filters,
            n_results=5,
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
        # Format the prompt
        context_str = "\n".join([f"- {src['content_snippet']}" for src in context])
        history_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]]
        )  # Last 5 messages

        prompt = (
            "You are a helpful assistant for exploring digital archives."
            "\n\n"
            "Conversation History:\n" + history_str + "\n\n"
            "Retrieved Context from Files:\n" + context_str + "\n\n"
            "User Question: " + query_text + "\n\n"
            "Please provide a helpful response based on the context and "
            "conversation history.\n"
            "If the context doesn't contain relevant information, say so clearly."
        )

        # Call Ollama with streaming
        response = ollama.chat(
            model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True
        )

        return response
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return iter([])  # Return empty iterator
