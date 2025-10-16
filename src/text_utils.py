import logging
import os
from pathlib import Path
from typing import Union

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PREFERRED_TOKENIZER = "meta-llama/Meta-Llama-3-8B"
FALLBACK_TOKENIZER = "bert-base-uncased"


def _load_tokenizer() -> AutoTokenizer:
    """Initialize tokenizer, preferring the LLaMA 3 vocabulary."""
    candidates: list[Union[str, Path]] = []

    env_path = os.getenv("LLAMA3_TOKENIZER_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            candidates.append(path)

    candidates.append(PREFERRED_TOKENIZER)
    candidates.append(FALLBACK_TOKENIZER)

    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=False)
            logger.debug("Loaded tokenizer from %s", candidate)
            return tokenizer
        except Exception as exc:
            logger.debug("Tokenizer load failed for %s: %s", candidate, exc)

    raise RuntimeError("Failed to initialize text tokenizer")


tokenizer = _load_tokenizer()


def chunk_text(text: str, max_tokens: int = 256) -> list[str]:
    """Split text into chunks of a specified max token size."""
    logger.debug("Tokenizing text of length %d for chunking.", len(text))

    # Tokenize the entire text to get token IDs
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Manually split the token list into chunks
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        # Decode the tokens back into a string
        chunks.append(tokenizer.decode(chunk_tokens))

    logger.debug("Split text into %d chunks.", len(chunks))
    return chunks
