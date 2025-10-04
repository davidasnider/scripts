"""AI-powered analysis utilities for file content."""

from __future__ import annotations

import base64
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


def analyze_financial_document(text: str) -> dict[str, Any]:
    """Analyze financial document text using an LLM as a forensic accountant.

    Parameters
    ----------
    text : str
        The financial document text to analyze.

    Returns
    -------
    dict[str, Any]
        A dictionary with 'summary', 'potential_red_flags', 'incriminating_items',
        and 'confidence_score'.
    """
    prompt = (
        "You are a meticulous forensic accountant. Analyze the following "
        "financial document text and provide a JSON response with exactly "
        "four keys:\n\n"
        '- "summary": a concise paragraph summarizing the document.\n'
        '- "potential_red_flags": a list of potential red flags or irregularities.\n'
        '- "incriminating_items": a list of items that could be incriminating.\n'
        '- "confidence_score": a numerical score from 0 to 100 indicating confidence '
        "in the analysis.\n\n"
        f"Text: {text}\n\n"
        "Respond only with valid JSON."
    )

    response = ollama.chat(
        model="deepseek-coder-v2:16b-lite-instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    json_str = response["message"]["content"]
    return json.loads(json_str)


def describe_image(image_path: str) -> str:
    """Generate a detailed description of an image using a vision-capable LLM.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    str
        Detailed description of the image.
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    response = ollama.chat(
        model="llava:34b-v1.6",
        messages=[
            {
                "role": "user",
                "content": "Describe this image in detail.",
                "images": [image_b64],
            }
        ],
    )
    return response["message"]["content"]
