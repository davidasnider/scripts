from __future__ import annotations

import json
from unittest.mock import patch

from httpx import ConnectError

from src.ai_analyzer import analyze_estate_relevant_information


def _build_response(payload: dict) -> dict:
    return {"message": {"content": json.dumps(payload)}}


@patch("src.ai_analyzer._ollama_chat")
def test_analyze_estate_single_chunk_extracts_information(
    mock_ollama_chat,
):
    sample_text = (
        "My will is kept in the blue folder in the study desk. "
        "Contact Attorney Lisa Grant at 555-1234 for probate."
    )
    mocked_payload = {
        "Legal": [
            {
                "item": "Last Will and Testament",
                "why_it_matters": "Identifies formal estate instructions",
                "details": "Stored in the blue folder in the study desk",
                "contact": "Attorney Lisa Grant, 555-1234",
            }
        ]
    }
    mock_ollama_chat.return_value = _build_response(mocked_payload)

    result = analyze_estate_relevant_information(sample_text, source_name="letter.txt")

    assert result["has_estate_relevant_info"] is True
    assert "Legal" in result["estate_information"]
    assert result["estate_information"]["Legal"][0]["item"] == "Last Will and Testament"
    assert result["_chunk_count"] == 1
    mock_ollama_chat.assert_called_once()


@patch("src.ai_analyzer.chunk_text", return_value=["chunk-1", "chunk-2"])
@patch("src.ai_analyzer._ollama_chat")
def test_analyze_estate_multi_chunk_merges_results(mock_ollama_chat, _mock_chunk_text):
    long_text = "x" * 4005
    chunk_payloads = [
        {
            "Financial": [
                {
                    "item": "Chase Savings Account",
                    "why_it_matters": "Access funds for estate expenses",
                    "details": "Account ending in 4221, branch on 5th Ave",
                }
            ]
        },
        {
            "Digital": [
                {
                    "item": "Email Account",
                    "why_it_matters": "Communicate with contacts and reset passwords",
                    "details": (
                        "Primary email at example@example.com, stored passwords in "
                        "1Password"
                    ),
                }
            ]
        },
    ]
    side_effect = [_build_response(payload) for payload in chunk_payloads]
    mock_ollama_chat.side_effect = side_effect

    result = analyze_estate_relevant_information(
        long_text, source_name="vault-notes.txt"
    )

    assert result["has_estate_relevant_info"] is True
    assert "Financial" in result["estate_information"]
    assert "Digital" in result["estate_information"]
    assert len(result["estate_information"]["Financial"]) == 1
    assert len(result["estate_information"]["Digital"]) == 1
    assert result["_chunk_count"] == 2
    assert mock_ollama_chat.call_count == 2


@patch("src.ai_analyzer._ollama_chat")
def test_analyze_estate_returns_default_on_llm_failure(
    mock_ollama_chat,
):
    sample_text = "Funeral wishes are detailed in my notebook."
    mock_ollama_chat.side_effect = RuntimeError("LLM unavailable")

    result = analyze_estate_relevant_information(sample_text, source_name="notes.txt")

    assert result["has_estate_relevant_info"] is False
    assert result["estate_information"] == {}
    assert result["_chunk_count"] == 0


@patch("src.ai_analyzer._ollama_chat")
def test_analyze_estate_handles_connection_error(mock_ollama_chat):
    mock_ollama_chat.side_effect = ConnectError("Connection failed")

    result = analyze_estate_relevant_information(
        "test text", source_name="connect_error.txt"
    )

    assert result["has_estate_relevant_info"] is False
    assert result["estate_information"] == {}
    assert result["_chunk_count"] == 0
