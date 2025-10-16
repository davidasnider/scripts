"""AI-powered analysis utilities for file content."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import ollama
import yaml
from PIL import Image, UnidentifiedImageError

from src.text_utils import chunk_text

logger = logging.getLogger(__name__)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TEXT_ANALYZER_MODEL = config["models"]["text_analyzer"]
CODE_ANALYZER_MODEL = config["models"]["code_analyzer"]
IMAGE_DESCRIBER_MODEL = config["models"]["image_describer"]


def _clean_json_response(response_text: str) -> str:
    """Clean JSON response by removing code block markers if present."""
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]

    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return response_text.strip()


def analyze_text_content(text: str) -> dict[str, Any]:
    """Analyze text content using an LLM to extract summary and mentioned people."""
    text_bytes = len(text.encode("utf-8"))
    logger.debug(
        "Starting text content analysis, text length: %d bytes",
        text_bytes,
    )
    if text_bytes <= 3000:
        # Single chunk processing
        prompt = (
            "You are a document analyst. Analyze the following text and provide a "
            "JSON response with exactly two keys:\n\n"
            '- "summary": a concise paragraph summarizing the main points of the '
            "text.\n"
            '- "mentioned_people": a list of names of people mentioned in the text.\n\n'
            f"Text: {text}\n\n"
            "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
            "backticks. Return only the raw JSON object."
        )

        try:
            logger.debug(
                "Sending Ollama text analysis request (single chunk), "
                "prompt length: %d",
                len(prompt),
            )
            response = ollama.chat(
                model=TEXT_ANALYZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            logger.debug("Received Ollama response for text analysis")
            json_str = response["message"]["content"]
            return json.loads(_clean_json_response(json_str))
        except Exception as e:
            logger.warning("Failed to analyze text content with Ollama: %s", e)
            return {
                "summary": "Analysis unavailable - Ollama not accessible",
                "mentioned_people": [],
            }

    # Multi-chunk processing
    chunks = chunk_text(text, max_tokens=2048)
    all_summaries = []
    all_people = set()

    for i, chunk in enumerate(chunks):
        prompt = (
            f"You are a document analyst. Analyze chunk {i+1} of {len(chunks)} of the "
            "following text and provide a JSON response with exactly two keys:\n\n"
            '- "summary": a concise paragraph summarizing the main points of this '
            "chunk.\n"
            '- "mentioned_people": a list of names of people mentioned in this '
            "chunk.\n\n"
            f"Text chunk: {chunk}\n\n"
            "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
            "backticks. Return only the raw JSON object."
        )

        try:
            logger.debug(
                "Sending Ollama request for text chunk %d/%d, prompt length: %d",
                i + 1,
                len(chunks),
                len(prompt),
            )
            response = ollama.chat(
                model=TEXT_ANALYZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            logger.debug(
                "Received Ollama response for text chunk %d/%d", i + 1, len(chunks)
            )
            json_str = response["message"]["content"]
            chunk_result = json.loads(_clean_json_response(json_str))
            all_summaries.append(chunk_result.get("summary", ""))
            all_people.update(chunk_result.get("mentioned_people", []))
        except Exception as e:
            logger.warning("Failed to analyze text chunk %d with Ollama: %s", i + 1, e)
            continue

    if all_summaries:
        combined_summary_prompt = (
            "You are a document analyst. Combine the following chunk summaries into a "
            "single cohesive summary of the entire document:\n\n"
            "Chunk summaries:\n"
            + "\n\n".join(
                f"Chunk {i+1}: {summary}" for i, summary in enumerate(all_summaries)
            )
            + "\n\n"
            "Provide a concise paragraph summarizing the main points of the entire "
            "document. Your response should contain only the summary, with no "
            "introductory phrases like 'Here is a summary:'."
        )

        try:
            logger.debug(
                "Sending Ollama combine request for %d text summaries, "
                "prompt length: %d",
                len(all_summaries),
                len(combined_summary_prompt),
            )
            response = ollama.chat(
                model=TEXT_ANALYZER_MODEL,
                messages=[{"role": "user", "content": combined_summary_prompt}],
            )
            logger.debug("Received Ollama response for summary combination")
            final_summary = response["message"]["content"]
        except Exception as e:
            logger.warning("Failed to combine summaries with Ollama: %s", e)
            final_summary = " ".join(all_summaries)
    else:
        final_summary = "Analysis unavailable - Ollama not accessible"

    logger.debug("Completed text content analysis")
    return {
        "summary": final_summary,
        "mentioned_people": list(all_people),
    }


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
    text_bytes = len(text.encode("utf-8"))
    logger.debug(
        "Starting financial document analysis, text length: %d bytes",
        text_bytes,
    )
    # Check if text needs chunking (over 3000 bytes)
    if text_bytes <= 3000:
        # Single chunk processing
        prompt = (
            "You are a meticulous forensic accountant. Analyze the following "
            "financial document text and provide a JSON response with exactly "
            "four keys:\n\n"
            '- "summary": a concise paragraph summarizing the document.\n'
            '- "potential_red_flags": a list of potential red flags or '
            "irregularities.\n"
            '- "incriminating_items": a list of items that could be incriminating.\n'
            '- "confidence_score": a numerical score from 0 to 100 indicating '
            "confidence in the analysis.\n\n"
            f"Text: {text}\n\n"
            "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
            "backticks. Return only the raw JSON object."
        )

        try:
            logger.debug(
                "Sending Ollama financial analysis request (single chunk), "
                "prompt length: %d",
                len(prompt),
            )
            response = ollama.chat(
                model=CODE_ANALYZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"raw": True},
            )
            logger.debug("Received Ollama response for financial analysis")
            json_str = response["message"]["content"]
            return json.loads(_clean_json_response(json_str))
        except Exception as e:
            logger.warning("Failed to analyze financial document with Ollama: %s", e)
            return {
                "summary": "Analysis unavailable - Ollama not accessible",
                "potential_red_flags": [],
                "incriminating_items": [],
                "confidence_score": 0,
            }

    # Multi-chunk processing
    chunks = chunk_text(text, max_tokens=2048)
    all_summaries = []
    all_red_flags = set()
    all_incriminating = set()
    confidence_scores = []

    for i, chunk in enumerate(chunks):
        prompt = (
            f"You are a meticulous forensic accountant. Analyze chunk {i+1} of "
            f"{len(chunks)} of the following financial document text and provide a "
            "JSON response with exactly four keys:\n\n"
            '- "summary": a concise paragraph summarizing this chunk.\n'
            '- "potential_red_flags": a list of potential red flags or irregularities '
            "in this chunk.\n"
            '- "incriminating_items": a list of items that could be incriminating in '
            "this chunk.\n"
            '- "confidence_score": a numerical score from 0 to 100 indicating '
            "confidence in the analysis of this chunk.\n\n"
            f"Text chunk: {chunk}\n\n"
            "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
            "backticks. Return only the raw JSON object."
        )

        try:
            logger.debug(
                "Sending Ollama request for financial chunk %d/%d, prompt length: %d",
                i + 1,
                len(chunks),
                len(prompt),
            )
            response = ollama.chat(
                model=CODE_ANALYZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"raw": True},
            )
            logger.debug(
                "Received Ollama response for financial chunk %d/%d", i + 1, len(chunks)
            )
            json_str = response["message"]["content"]
            chunk_result = json.loads(_clean_json_response(json_str))
            all_summaries.append(chunk_result.get("summary", ""))
            all_red_flags.update(chunk_result.get("potential_red_flags", []))
            all_incriminating.update(chunk_result.get("incriminating_items", []))
            if isinstance(chunk_result.get("confidence_score"), (int, float)):
                confidence_scores.append(chunk_result["confidence_score"])
        except Exception as e:
            logger.warning(
                "Failed to analyze financial chunk %d with Ollama: %s", i + 1, e
            )
            continue

    # Combine results
    if all_summaries:
        combined_summary_prompt = (
            "You are a forensic accountant. Combine the following chunk summaries "
            "into a single cohesive summary of the entire financial document:\n\n"
            "Chunk summaries:\n"
            + "\n\n".join(
                f"Chunk {i+1}: {summary}" for i, summary in enumerate(all_summaries)
            )
            + "\n\n"
            "Provide a concise paragraph summarizing the main points of the entire "
            "financial document. Your response should contain only the summary, with "
            "no introductory phrases like 'Here is a summary:'."
        )

        try:
            logger.debug(
                "Sending Ollama combine request for %d financial summaries, "
                "prompt length: %d",
                len(all_summaries),
                len(combined_summary_prompt),
            )
            response = ollama.chat(
                model=CODE_ANALYZER_MODEL,
                messages=[{"role": "user", "content": combined_summary_prompt}],
                options={"raw": True},
            )
            logger.debug("Received Ollama response for financial summary combination")
            final_summary = response["message"]["content"]
        except Exception as e:
            logger.warning("Failed to combine financial summaries with Ollama: %s", e)
            final_summary = " ".join(all_summaries)
    else:
        final_summary = "Analysis unavailable - Ollama not accessible"

    # Calculate average confidence score
    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    )

    logger.debug("Completed financial document analysis")
    return {
        "summary": final_summary,
        "potential_red_flags": list(all_red_flags),
        "incriminating_items": list(all_incriminating),
        "confidence_score": int(avg_confidence),
    }


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
    logger.debug("Starting image description for %s", image_path)

    try:
        with Image.open(image_path) as handle:
            handle.verify()
    except UnidentifiedImageError as exc:
        logger.info(
            "Skipping image description for unsupported image %s: %s",
            image_path,
            exc,
        )
        return "Image description unavailable - unsupported image format"
    except Exception as exc:
        logger.info(
            "Skipping image description for unreadable image %s: %s",
            image_path,
            exc,
        )
        return "Image description unavailable - unreadable image"

    with open(image_path, "rb") as f:
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    try:
        logger.debug(
            "Sending Ollama request for image description, image size: %d bytes",
            len(image_b64),
        )
        response = ollama.chat(
            model=IMAGE_DESCRIBER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image in detail.",
                    "images": [image_b64],
                }
            ],
        )
        logger.debug("Received Ollama response for image description")
        logger.debug("Completed image description")
        return response["message"]["content"]
    except Exception as e:
        logger.warning("Failed to describe image with Ollama: %s", e)
        return "Image description unavailable - Ollama not accessible"


def summarize_video_frames(frame_descriptions: list[str]) -> str:
    """Summarize multiple frame descriptions into a cohesive video summary.

    Parameters
    ----------
    frame_descriptions : list[str]
        List of descriptions for each video frame.

    Returns
    -------
    str
        A summary of the video content based on frame descriptions.
    """
    logger.debug(
        "Starting video frame summarization for %d frames", len(frame_descriptions)
    )

    if not frame_descriptions:
        return "No frame descriptions available"

    if len(frame_descriptions) == 1:
        return frame_descriptions[0]

    # Combine frame descriptions
    combined_descriptions = "\n\n".join(
        f"Frame {i+1}: {desc}" for i, desc in enumerate(frame_descriptions)
    )

    prompt = (
        "You are a video analyst. Analyze the following frame descriptions from a "
        "video and provide a cohesive summary of the video's content. Focus on the "
        "main subjects, activities, and overall narrative shown in the frames.\n\n"
        f"Frame descriptions:\n{combined_descriptions}\n\n"
        "Provide a concise paragraph summarizing the video content. Your "
        "response should contain only the summary, with no introductory phrases like "
        "'Here is a summary:'."
    )

    try:
        logger.debug(
            "Sending Ollama request for video summarization, prompt length: %d",
            len(prompt),
        )
        response = ollama.chat(
            model=TEXT_ANALYZER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        logger.debug("Received Ollama response for video summarization")
        logger.debug("Completed video frame summarization")
        return response["message"]["content"]
    except Exception as e:
        logger.warning("Failed to summarize video frames with Ollama: %s", e)
        # Fallback: return a simple concatenation
        return "Video summary: " + " ".join(frame_descriptions[:3]) + "..."
