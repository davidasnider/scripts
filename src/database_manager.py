"""Database management utilities for ChromaDB and embeddings."""

from __future__ import annotations

import json

import chromadb
import ollama


def initialize_db(path: str):
    """Initialize a persistent ChromaDB client and return the digital_archive
    collection.

    Parameters
    ----------
    path : str
        Path to the persistent database directory.

    Returns
    -------
    chromadb.Collection
        The digital_archive collection.
    """
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name="digital_archive")
    return collection


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for the given text using nomic-embed-text.

    Parameters
    ----------
    text : str
        The text to embed.

    Returns
    -------
    list[float]
        The embedding vector.
    """
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def add_file_to_db(file_data: dict, collection) -> None:
    """Add a file's data to the ChromaDB collection with embedding and metadata.

    Parameters
    ----------
    file_data : dict
        The complete file data dictionary from the manifest.
    collection : chromadb.Collection
        The ChromaDB collection to add to.
    """
    # Consolidate relevant text
    text_parts = [
        file_data.get("file_name", ""),
        file_data.get("summary", ""),
        file_data.get("description", ""),
        file_data.get("extracted_text", ""),
        ", ".join(file_data.get("mentioned_people", [])),
    ]
    consolidated_text = " ".join(text_parts).strip()

    # Generate embedding
    embedding = generate_embedding(consolidated_text)

    # Structured metadata for filtering
    metadata = {
        "file_type": file_data.get("mime_type", "").split("/")[
            0
        ],  # e.g., 'image', 'text'
        "is_nsfw": file_data.get("is_nsfw", False),
        "has_financial_red_flags": file_data.get("has_financial_red_flags", False),
    }

    # Add to collection
    collection.add(
        ids=[file_data["file_path"]],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[json.dumps(file_data)],
    )
