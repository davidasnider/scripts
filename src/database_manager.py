"""Database management utilities for ChromaDB and embeddings."""

from __future__ import annotations

import json
import logging

import chromadb
import numpy as np
import ollama

from src.config_utils import (
    build_ollama_options,
    compute_chunk_size,
    get_model_config,
    load_config,
)
from src.text_utils import chunk_text, count_tokens

logger = logging.getLogger(__name__)

# Load config
config = load_config()
models_config = config.get("models", {})
embedding_model_config = get_model_config(models_config, "embedding_model")

EMBEDDING_MODEL = embedding_model_config["name"]
EMBEDDING_CONTEXT_WINDOW = embedding_model_config.get("context_window")
EMBEDDING_OPTIONS = build_ollama_options(embedding_model_config)
EMBEDDING_PROMPT_RESERVE = 512
EMBEDDING_TOKEN_MARGIN_FACTOR = 1.8
EMBEDDING_CHUNK_TOKENS = compute_chunk_size(
    EMBEDDING_CONTEXT_WINDOW,
    reserve_tokens=EMBEDDING_PROMPT_RESERVE,
    safety_ratio=0.4,
)
logger.debug(
    "Embedding model configured: %s (context_window=%s, chunk_tokens=%s)",
    EMBEDDING_MODEL,
    EMBEDDING_CONTEXT_WINDOW,
    EMBEDDING_CHUNK_TOKENS,
)
DEFAULT_EMBEDDING_DIM = 768


def initialize_db(path: str):
    """Initialize a persistent ChromaDB client and return the digital_archive
    collection."""
    logger.info("Initializing ChromaDB at path: %s", path)
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name="digital_archive")
    logger.info("ChromaDB initialized successfully")
    return collection


def _prepare_embedding_chunks(text: str, initial_limit: int) -> tuple[list[str], int]:
    """Chunk text for embeddings while respecting the model context window."""

    limit = max(int(initial_limit / EMBEDDING_TOKEN_MARGIN_FACTOR), 128)

    while True:
        chunks = chunk_text(text, max_tokens=limit)
        if not chunks:
            return [], limit

        if EMBEDDING_CONTEXT_WINDOW is None:
            return chunks, limit

        fits = True
        for chunk in chunks:
            token_count = count_tokens(chunk)
            if token_count > EMBEDDING_CONTEXT_WINDOW:
                fits = False
                break
            if (
                int(token_count * EMBEDDING_TOKEN_MARGIN_FACTOR)
                > EMBEDDING_CONTEXT_WINDOW
            ):
                fits = False
                break

        if fits:
            return chunks, limit

        new_limit = max(limit // 2, 128)
        if new_limit == limit:
            return chunks, limit
        limit = new_limit


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for the given text using the configured model.
    If the text is long, it will be chunked and the embeddings will be averaged.
    """
    logger.debug("Generating embedding for text of length %d", len(text))

    limit = EMBEDDING_CHUNK_TOKENS
    min_limit = 128

    while limit >= min_limit:
        chunks, chunk_limit = _prepare_embedding_chunks(text, limit)
        if not chunks:
            logger.debug("No chunks generated for embedding (limit=%d)", chunk_limit)
            break

        if chunk_limit != limit:
            logger.debug(
                "Adjusted embedding chunk limit from %d to %d to fit context hint",
                limit,
                chunk_limit,
            )
            limit = chunk_limit

        embeddings = []
        context_failure = False

        for chunk in chunks:
            try:
                logger.debug(
                    "Sending Ollama embedding request for chunk of length %d",
                    len(chunk),
                )
                request_kwargs = {"model": EMBEDDING_MODEL, "prompt": chunk}
                if EMBEDDING_OPTIONS:
                    request_kwargs["options"] = dict(EMBEDDING_OPTIONS)
                response = ollama.embeddings(**request_kwargs)
                embeddings.append(response["embedding"])
                logger.debug("Received Ollama embedding response for chunk")
            except Exception as exc:
                error_text = str(exc)
                context_failure = True
                if "context length" in error_text.lower():
                    logger.warning(
                        (
                            "Embedding chunk exceeded context window (chunk_limit=%d, "
                            "context=%s). Reducing chunk size and retrying."
                        ),
                        chunk_limit,
                        EMBEDDING_CONTEXT_WINDOW,
                    )
                else:
                    logger.warning(
                        (
                            "Failed to generate embedding for chunk with Ollama: %s. "
                            "Retrying with smaller chunk size."
                        ),
                        exc,
                    )
                break

        if context_failure:
            new_limit = max(limit // 2, min_limit)
            if new_limit >= limit:
                new_limit = max(limit - 256, min_limit)

            if new_limit == limit:
                logger.warning(
                    "Unable to reduce embedding chunk size further (limit=%d); "
                    "using fallback zero vector.",
                    limit,
                )
                break

            logger.debug(
                "Retrying embeddings with reduced chunk limit (old=%d, new=%d)",
                limit,
                new_limit,
            )
            limit = new_limit
            continue

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            logger.debug(
                "Successfully generated and averaged embeddings using %d chunk(s)",
                len(embeddings),
            )
            return avg_embedding

        logger.warning(
            "No embeddings generated for text with current chunk limit (%d). "
            "Reducing and retrying.",
            limit,
        )
        limit = max(limit // 2, min_limit)

    logger.warning(
        "Failed to generate embeddings after chunk limit adjustments; "
        "returning fallback zero vector."
    )
    return [0.0] * DEFAULT_EMBEDDING_DIM


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
        (
            json.dumps(file_data.get("estate_information", {}), sort_keys=True)
            if file_data.get("estate_information")
            else ""
        ),
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
        "has_estate_relevant_info": file_data.get("has_estate_relevant_info") or False,
    }

    # Add to collection
    collection.add(
        ids=[file_data["file_path"]],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[consolidated_text],
    )
    logger.debug("Successfully added file to DB")
