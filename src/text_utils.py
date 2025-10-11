import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Load a tokenizer for a BERT-based model
# 'bert-base-uncased' is a good proxy for many embedding models
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


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
