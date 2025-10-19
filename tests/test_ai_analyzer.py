import json
from unittest.mock import patch

from src.ai_analyzer import analyze_text_content, detect_passwords


@patch("src.ai_analyzer.ollama.chat")
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


def test_detect_passwords_returns_default_for_empty_text():
    """Empty or whitespace-only text should yield a negative password result."""
    assert detect_passwords("   ") == {"contains_password": False, "passwords": {}}


@patch("src.ai_analyzer.ollama.chat")
def test_detect_passwords_single_chunk(mock_ollama_chat):
    """Verify password detection parses single-chunk responses correctly."""
    mock_ollama_chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "contains_password": True,
                    "passwords": {"admin_password": "s3cr3t!"},
                }
            )
        }
    }

    text = "Admin credentials:\npassword: s3cr3t!"
    result = detect_passwords(text, source_name="credentials.txt")

    assert result["contains_password"] is True
    assert result["passwords"] == {"admin_password": "s3cr3t!"}
    mock_ollama_chat.assert_called_once()


@patch("src.ai_analyzer.chunk_text", return_value=["chunk-one", "chunk-two"])
@patch("src.ai_analyzer.ollama.chat")
def test_detect_passwords_multi_chunk_deduplicates_keys(mock_ollama_chat, _mock_chunk):
    """Ensure multi-chunk responses merge password dictionaries safely."""

    mock_responses = [
        {"contains_password": True, "passwords": {"admin": "secret1"}},
        {
            "contains_password": True,
            "passwords": {"admin": "secret2", "backup": "b@ckup"},
        },
    ]

    def _chat_side_effect(*args, **kwargs):
        # Pop responses in order
        content = mock_responses.pop(0)
        return {"message": {"content": json.dumps(content)}}

    mock_ollama_chat.side_effect = _chat_side_effect

    long_text = "A" * 4000  # Force multi-chunk path
    result = detect_passwords(long_text, source_name="long.txt")

    assert result["contains_password"] is True
    assert result["passwords"] == {
        "admin": "secret1",
        "admin_2": "secret2",
        "backup": "b@ckup",
    }
    assert mock_ollama_chat.call_count == 2
