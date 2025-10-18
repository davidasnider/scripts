"""Database management utilities for ChromaDB and embeddings."""

from __future__ import annotations

import logging

import chromadb
import numpy as np
import ollama
import yaml

from src.text_utils import chunk_text

logger = logging.getLogger(__name__)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL = config["models"]["embedding_model"]
DEFAULT_EMBEDDING_DIM = 768


def initialize_db(path: str):
    """Initialize a persistent ChromaDB client and return the digital_archive
    collection."""
    logger.info("Initializing ChromaDB at path: %s", path)
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name="digital_archive")
    logger.info("ChromaDB initialized successfully")
    return collection


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for the given text using the configured model.
    If the text is long, it will be chunked and the embeddings will be averaged.
    """
    logger.debug("Generating embedding for text of length %d", len(text))

    chunks = chunk_text(text)
    embeddings = []

    for chunk in chunks:
        try:
            logger.debug(
                "Sending Ollama embedding request for chunk of length %d", len(chunk)
            )
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
            embeddings.append(response["embedding"])
            logger.debug("Received Ollama embedding response for chunk")
        except Exception as e:
            logger.warning("Failed to generate embedding for chunk with Ollama: %s", e)
            continue

    if not embeddings:
        logger.warning("No embeddings were generated for the text.")
        return [0.0] * DEFAULT_EMBEDDING_DIM  # Fallback zero vector

    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0).tolist()
    logger.debug("Successfully generated and averaged embeddings")
    return avg_embedding


def add_file_to_db(file_data: dict, collection) -> None:
    """Add a file's data to the ChromaDB collection with embedding and metadata.

    Parameters
    ----------
    file_data : dict
        The complete file data dictionary from the manifest.
    collection : chromadb.Collection
        The ChromaDB collection to add to.
    """
    logger.debug("Adding file to DB: %s", file_data.get("file_path", ""))
    # Consolidate relevant text
    text_parts = [
        file_data.get("file_name", ""),
        file_data.get("summary", ""),
        file_data.get("description", ""),
        file_data.get("extracted_text", ""),
        ", ".join(file_data.get("mentioned_people", [])),
    ]
    consolidated_text = " ".join(part for part in text_parts if part).strip()

    # Generate embedding only if there is text
    if consolidated_text:
        embedding = generate_embedding(consolidated_text)
    else:
        # Use a zero vector if there is no text to embed
        embedding = [0.0] * DEFAULT_EMBEDDING_DIM

    # Structured metadata for filtering
    metadata = {
        "file_path": file_data.get("file_path") or "",
        "file_name": file_data.get("file_name") or "",
        "mime_type": file_data.get("mime_type") or "",
        "file_type": (file_data.get("mime_type") or "").split("/")[0],
        "is_nsfw": file_data.get("is_nsfw") or False,
        "has_financial_red_flags": file_data.get("has_financial_red_flags") or False,
    }

    # Add to collection
    collection.add(
        ids=[file_data["file_path"]],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[consolidated_text],
    )
    logger.debug("Successfully added file to DB")
