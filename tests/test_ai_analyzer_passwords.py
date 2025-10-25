import json
from unittest.mock import patch

from src.ai_analyzer import detect_passwords, DEFAULT_PASSWORD_RESULT

# Common mock for a successful ollama chat response
def mock_ollama_chat_response(content: dict):
    return {"message": {"content": json.dumps(content)}}

def test_detect_passwords_returns_default_for_empty_text():
    """Empty or whitespace-only text should yield a negative password result."""
    assert detect_passwords("   ") == DEFAULT_PASSWORD_RESULT


@patch("src.ai_analyzer._ollama_chat")
def test_detect_passwords_handles_clean_json_response(mock_ollama_chat):
    """Test that _normalize_result correctly handles a clean JSON response."""
    mock_response = {
        "passwords": [
            {"context": "user login", "password": "password123"},
        ]
    }
    mock_ollama_chat.return_value = mock_ollama_chat_response(mock_response)

    result = detect_passwords("some text")

    assert result["contains_password"] is True
    assert result["passwords"] == [{"context": "user login", "password": "password123"}]

@patch("src.ai_analyzer._ollama_chat")
def test_detect_passwords_handles_malformed_json_response(mock_ollama_chat):
    """Test that _normalize_result filters out malformed entries from the AI response."""
    mock_response = {
        "passwords": [
            {"context": "user login", "password": "password123"},
            {"context": "missing password"},
            {"password": "missing context"},
            "just a string",
        ]
    }
    mock_ollama_chat.return_value = mock_ollama_chat_response(mock_response)

    result = detect_passwords("some text")

    assert result["contains_password"] is True
    assert result["passwords"] == [{"context": "user login", "password": "password123"}]

@patch("src.ai_analyzer._ollama_chat")
def test_detect_passwords_identifies_complex_password(mock_ollama_chat):
    """Test that a complex, password-manager-style string is correctly identified."""
    mock_response = {
        "passwords": [
            {"context": "Password Manager", "password": "aBc-123-!@#"},
        ]
    }
    mock_ollama_chat.return_value = mock_ollama_chat_response(mock_response)

    text = "Here is my password: aBc-123-!@#"
    result = detect_passwords(text)

    assert result["contains_password"] is True
    assert result["passwords"] == [{"context": "Password Manager", "password": "aBc-123-!@#"}]

@patch("src.ai_analyzer._ollama_chat")
def test_detect_passwords_identifies_api_key(mock_ollama_chat):
    """Test that a machine-generated API key is correctly identified."""
    mock_response = {
        "passwords": [
            {"context": "API Key", "password": "xyz-apikey-12345"},
        ]
    }
    mock_ollama_chat.return_value = mock_ollama_chat_response(mock_response)

    text = "My API key for authentication is xyz-apikey-12345"
    result = detect_passwords(text)

    assert result["contains_password"] is True
    assert result["passwords"] == [{"context": "API Key", "password": "xyz-apikey-12345"}]
