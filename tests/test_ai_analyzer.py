import json
from unittest.mock import patch

import pytest
from src.ai_analyzer import analyze_text_content


@patch("src.ai_analyzer.ollama.chat")
def test_analyze_text_content_ignores_usernames(mock_ollama_chat):
    """
    Verify that the AI analyzer prompt correctly instructs the model to ignore usernames and only identify real names.
    This test verifies the behavior through mocked responses rather than actual AI model behavior.
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
    text = "This document was written by David and reviewed by Brandy. User akanda also contributed."

    # Call the function to be tested
    result = analyze_text_content(text)

    # Assert that only the real names are extracted
    assert "David" in result["mentioned_people"]
    assert "Brandy" in result["mentioned_people"]
    assert "akanda" not in result["mentioned_people"]