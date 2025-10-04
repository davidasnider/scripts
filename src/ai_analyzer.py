"""AI-powered analysis utilities for file content."""

from __future__ import annotations

import json
from typing import Any

import ollama


def analyze_text_content(text: str) -> dict[str, Any]:
    """Analyze text content using an LLM to extract summary and mentioned people.

    Parameters
    ----------
    text : str
        The text content to analyze.

    Returns
    -------
    dict[str, Any]
        A dictionary with 'summary' (str) and 'mentioned_people' (list[str]).
    """
    prompt = (
        "You are a document analyst. Analyze the following text and provide a "
        "JSON response with exactly two keys:\n\n"
        '- "summary": a concise paragraph summarizing the main points of the text.\n'
        '- "mentioned_people": a list of names of people mentioned in the text.\n\n'
        f"Text: {text}\n\n"
        "Respond only with valid JSON."
    )

    response = ollama.chat(
        model="llama3:70b-instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    json_str = response["message"]["content"]
    return json.loads(json_str)
