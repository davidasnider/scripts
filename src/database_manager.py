"""Database management utilities for ChromaDB and embeddings."""

from __future__ import annotations

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
