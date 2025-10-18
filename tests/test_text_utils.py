import src.text_utils as text_utils
from src.text_utils import chunk_text


def test_chunk_text_splits_correctly():
    """Verify that text is split into correctly sized chunks."""
    text = "This is a long string of text that needs to be split into several chunks."
    max_tokens = 5
    chunks = chunk_text(text, max_tokens=max_tokens)

    # Verify that each chunk's token count is within the specified limit
    for chunk in chunks:
        token_ids = text_utils.tokenizer.encode(chunk, add_special_tokens=False)
        assert len(token_ids) <= max_tokens

    # Verify that the reconstructed text from chunks matches the original text,
    # ignoring minor differences from tokenization/detokenization.
    reconstructed_text = "".join(chunks)
    assert reconstructed_text.lower().replace(" ", "") == text.lower().replace(" ", "")
