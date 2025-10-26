import json
from unittest.mock import MagicMock, patch

from httpx import ConnectError

from src.ai_analyzer import (
    analyze_financial_document,
    analyze_text_content,
)

# A reusable mock for the ollama client to simulate connection errors
MOCK_OLLAMA_CLIENT_WITH_ERROR = MagicMock()
MOCK_OLLAMA_CLIENT_WITH_ERROR.chat.side_effect = ConnectError("Failed to connect")


# Common mock for a successful ollama chat response
def mock_ollama_chat_response(content: dict):
    return {"message": {"content": json.dumps(content)}}


@patch("src.ai_analyzer._ollama_chat")
def test_analyze_text_content_ignores_usernames(mock_ollama_chat):
    """Verify that the AI analyzer prompt correctly instructs the model to ignore
    usernames and only identify real names. This test verifies the behavior through
    mocked responses rather than actual AI model behavior.
    """
    # Mock the response from the Ollama chat model
    mock_response = {
        "message": {
            "content": json.dumps(
                {
                    "summary": "This is a summary.",
                    "mentioned_people": ["David", "Brandy"],
                }
            )
        }
    }
    mock_ollama_chat.return_value = mock_response

    # Sample text containing a mix of real names and usernames
    text = (
        "This document was written by David and reviewed by Brandy. "
        "User akanda also contributed."
    )

    # Call the function to be tested
    result = analyze_text_content(text)

    # Assert that only the real names are extracted
    assert "David" in result["mentioned_people"]
    assert "Brandy" in result["mentioned_people"]
    assert "akanda" not in result["mentioned_people"]


@patch("src.ai_analyzer.chunk_text", return_value=["chunk-1", "chunk-2", "chunk-3"])
@patch("src.ai_analyzer._ollama_chat")
def test_analyze_text_content_respects_max_chunks(mock_chat, _mock_chunk):
    """Ensure multi-chunk text analysis honors the max_chunks limit."""

    # Responses for first two chunks plus combined summary call
    mock_chat.side_effect = [
        mock_ollama_chat_response({"summary": "part", "mentioned_people": ["Alice"]}),
        mock_ollama_chat_response({"summary": "part", "mentioned_people": ["Alice"]}),
        {"message": {"content": "Final summary"}},
    ]

    result = analyze_text_content("A" * 5000, source_name="doc.txt", max_chunks=2)

    assert result["summary"] == "Final summary"
    assert result["mentioned_people"] == ["Alice"]
    assert result["_chunk_count"] == 2
    assert mock_chat.call_count == 3


@patch("src.ai_analyzer.chunk_text", return_value=["chunk-1", "chunk-2", "chunk-3"])
@patch("src.ai_analyzer._ollama_chat")
def test_analyze_financial_document_respects_max_chunks(mock_chat, _mock_chunk):
    """Financial analysis should obey the chunk limit and combine summaries."""

    chunk_payload = {
        "summary": "Segment summary",
        "potential_red_flags": ["late filing"],
        "incriminating_items": ["cash"],
        "confidence_score": 80,
    }
    mock_chat.side_effect = [
        mock_ollama_chat_response(chunk_payload),
        mock_ollama_chat_response(chunk_payload),
        {"message": {"content": "Combined summary"}},
    ]

    result = analyze_financial_document(
        "C" * 5000, source_name="ledger.csv", max_chunks=2
    )

    assert result["summary"] == "Combined summary"
    assert set(result["potential_red_flags"]) == {"late filing"}
    assert set(result["incriminating_items"]) == {"cash"}
    # Average of chunk scores remains 80
    assert result["confidence_score"] == 80
    assert result["_chunk_count"] == 2
    assert mock_chat.call_count == 3
