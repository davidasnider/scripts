from __future__ import annotations

import json
from unittest.mock import patch

from src.ai_analyzer import analyze_estate_relevant_information


def _build_response(payload: dict) -> dict:
    return {"message": {"content": json.dumps(payload)}}


def test_analyze_estate_single_chunk_extracts_information():
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

    with patch(
        "src.ai_analyzer.ollama.chat", return_value=_build_response(mocked_payload)
    ) as mock_chat:
        result = analyze_estate_relevant_information(
            sample_text, source_name="letter.txt"
        )

    assert result["has_estate_relevant_info"] is True
    assert "Legal" in result["estate_information"]
    assert result["estate_information"]["Legal"][0]["item"] == "Last Will and Testament"
    mock_chat.assert_called_once()


def test_analyze_estate_multi_chunk_merges_results():
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

    with (
        patch("src.ai_analyzer.chunk_text", return_value=["chunk-1", "chunk-2"]),
        patch(
            "src.ai_analyzer.ollama.chat",
            side_effect=[_build_response(payload) for payload in chunk_payloads],
        ) as mock_chat,
    ):
        result = analyze_estate_relevant_information(
            long_text, source_name="vault-notes.txt"
        )

    assert result["has_estate_relevant_info"] is True
    assert "Financial" in result["estate_information"]
    assert "Digital" in result["estate_information"]
    assert len(result["estate_information"]["Financial"]) == 1
    assert len(result["estate_information"]["Digital"]) == 1
    assert mock_chat.call_count == 2


def test_analyze_estate_returns_default_on_llm_failure():
    sample_text = "Funeral wishes are detailed in my notebook."

    with patch(
        "src.ai_analyzer.ollama.chat", side_effect=RuntimeError("LLM unavailable")
    ):
        result = analyze_estate_relevant_information(
            sample_text, source_name="notes.txt"
        )

    assert result["has_estate_relevant_info"] is False
    assert result["estate_information"] == {}
