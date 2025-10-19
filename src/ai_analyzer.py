"""AI-powered analysis utilities for file content."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Callable

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
# Use TEXT_ANALYZER_MODEL as default if no password detector model in config.
PASSWORD_DETECTOR_MODEL = config["models"].get("password_detector", TEXT_ANALYZER_MODEL)

DEFAULT_PASSWORD_RESULT = {"contains_password": False, "passwords": {}}
DEFAULT_ESTATE_RESULT = {
    "has_estate_relevant_info": False,
    "estate_information": {},
}
ESTATE_CATEGORY_KEYS = [
    "Legal",
    "Financial",
    "Insurance",
    "Digital",
    "Medical",
    "Personal",
]


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


def _resolve_source_name(source_name: str | None) -> str:
    """Return a display-friendly source name for logging."""
    if not source_name:
        return "unknown-source"
    try:
        resolved = Path(source_name).name
        return resolved or source_name
    except (TypeError, ValueError):
        return str(source_name)


AbortCallback = Callable[[], None]


def _maybe_abort(callback: AbortCallback | None) -> None:
    if callback is not None:
        callback()


def _limit_chunk_list(
    chunks: list[str],
    max_chunks: int | None,
    *,
    analysis_name: str,
    source_name: str,
) -> list[str]:
    """Limit chunked text to at most max_chunks entries."""

    if max_chunks and max_chunks > 0 and len(chunks) > max_chunks:
        logger.info(
            "Limiting %s on %s to first %d chunk(s) (from %d).",
            analysis_name,
            source_name,
            max_chunks,
            len(chunks),
        )
        return chunks[:max_chunks]
    return chunks


def analyze_text_content(
    text: str,
    *,
    source_name: str | None = None,
    should_abort: AbortCallback | None = None,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Analyze text content using an LLM to extract summary and mentioned people."""
    text_bytes = len(text.encode("utf-8"))
    source_display_name = _resolve_source_name(source_name)
    logger.debug(
        "Starting text content analysis, text length: %d bytes",
        text_bytes,
    )
    if text_bytes <= 3000:
        _maybe_abort(should_abort)
        logger.info("Text analysis processing single chunk for %s", source_display_name)
        # Single chunk processing
        prompt = (
            "You are a document analyst. Analyze the following text and provide a "
            "JSON response with exactly two keys:\n\n"
            '- "summary": a concise paragraph summarizing the main points of the '
            "text.\n"
            '- "mentioned_people": a list of names of people mentioned in the text. '
            "These should be actual names, like 'John Smith' or 'Maria Garcia', "
            "not usernames like 'user123'.\n\n"
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
            _maybe_abort(should_abort)
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
    chunks = _limit_chunk_list(
        chunks,
        max_chunks,
        analysis_name="text analysis",
        source_name=source_display_name,
    )
    total_chunks = len(chunks)
    logger.info(
        "Text analysis processing %d chunk(s) for %s",
        total_chunks,
        source_display_name,
    )
    all_summaries = []
    all_people = set()

    for i, chunk in enumerate(chunks):
        _maybe_abort(should_abort)
        remaining_chunks = total_chunks - (i + 1)
        logger.info(
            "Text analysis chunk %d/%d for %s (%d remaining)",
            i + 1,
            total_chunks,
            source_display_name,
            remaining_chunks,
        )
        prompt = (
            f"You are a document analyst. Analyze chunk {i+1} of {total_chunks} of the "
            "following text and provide a JSON response with exactly two keys:\n\n"
            '- "summary": a concise paragraph summarizing the main points of this '
            "chunk.\n"
            '- "mentioned_people": a list of names of people mentioned in this '
            "chunk. These should be actual names, like 'John Smith' or "
            "'Maria Garcia', not usernames like 'user123'.\n\n"
            f"Text chunk: {chunk}\n\n"
            "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
            "backticks. Return only the raw JSON object."
        )

        try:
            logger.debug(
                "Sending Ollama request for text chunk %d/%d for %s, prompt length: %d",
                i + 1,
                total_chunks,
                source_display_name,
                len(prompt),
            )
            _maybe_abort(should_abort)
            response = ollama.chat(
                model=TEXT_ANALYZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            logger.debug(
                "Received Ollama response for text chunk %d/%d for %s",
                i + 1,
                total_chunks,
                source_display_name,
            )
            json_str = response["message"]["content"]
            chunk_result = json.loads(_clean_json_response(json_str))
            all_summaries.append(chunk_result.get("summary", ""))
            all_people.update(chunk_result.get("mentioned_people", []))
        except Exception as e:
            logger.warning(
                "Failed to analyze text chunk %d for %s with Ollama: %s",
                i + 1,
                source_display_name,
                e,
            )
            continue

    if all_summaries:
        _maybe_abort(should_abort)
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
            _maybe_abort(should_abort)
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


def detect_passwords(
    text: str,
    *,
    source_name: str | None = None,
    should_abort: AbortCallback | None = None,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Detect potential passwords within text using an LLM."""

    if not text.strip():
        return DEFAULT_PASSWORD_RESULT

    text_bytes = len(text.encode("utf-8"))
    source_display_name = _resolve_source_name(source_name)
    logger.debug(
        "Starting password detection, text length: %d bytes",
        text_bytes,
    )

    def _normalize_result(raw_result: dict[str, Any]) -> dict[str, Any]:
        contains_password = bool(raw_result.get("contains_password"))
        passwords_raw = raw_result.get("passwords") or {}
        if not isinstance(passwords_raw, dict):
            passwords: dict[str, str] = {}
        else:
            passwords = {
                str(key): str(value)
                for key, value in passwords_raw.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        if contains_password and not passwords:
            contains_password = False
        return {
            "contains_password": contains_password,
            "passwords": passwords,
        }

    prompt_template = (
        "You are a security auditor. Review the following text and determine "
        "whether it contains any strings that appear to be passwords, API keys, "
        "or other secret credentials.\n\n"
        "Respond with a JSON object using these keys:\n"
        '  - "contains_password": true if you see at least one likely password, '
        "otherwise false.\n"
        '  - "passwords": an object where each key is a short identifier '
        '(for example, the field label or "password_1") and each value is the '
        "exact password string taken from the text. Use an empty object when no "
        "passwords are detected.\n\n"
        "Only consider information present in the text. Do not invent entries. "
        "Respond with raw JSON only."
    )

    if text_bytes <= 3000:
        prompt = f"{prompt_template}\n\nText:\n{text}"
        try:
            _maybe_abort(should_abort)
            logger.debug(
                "Sending Ollama password detection request (single chunk) for %s",
                source_display_name,
            )
            response = ollama.chat(
                model=PASSWORD_DETECTOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            logger.debug("Received Ollama response for password detection")
            json_str = response["message"]["content"]
            raw_result = json.loads(_clean_json_response(json_str))
            return _normalize_result(raw_result)
        except Exception as e:
            logger.warning(
                "Failed to detect passwords in %s with Ollama: %s",
                source_display_name,
                e,
            )
            return DEFAULT_PASSWORD_RESULT

    chunks = chunk_text(text, max_tokens=2048)
    chunks = _limit_chunk_list(
        chunks,
        max_chunks,
        analysis_name="password detection",
        source_name=source_display_name,
    )
    total_chunks = len(chunks)
    detected_passwords: dict[str, str] = {}
    any_passwords = False

    def _deduplicate_key(key: str, value: str, existing: dict[str, str]) -> str:
        """Generate unique key for password to avoid collisions."""
        unique_key = key
        suffix = 1
        while unique_key in existing and existing[unique_key] != value:
            suffix += 1
            unique_key = f"{key}_{suffix}"
        return unique_key

    for i, chunk in enumerate(chunks):
        _maybe_abort(should_abort)
        remaining_chunks = total_chunks - (i + 1)
        logger.info(
            "Password detection chunk %d/%d for %s (%d remaining)",
            i + 1,
            total_chunks,
            source_display_name,
            remaining_chunks,
        )
        prompt = (
            f"{prompt_template}\n\n" f"This is chunk {i+1} of {total_chunks}:\n{chunk}"
        )
        try:
            response = ollama.chat(
                model=PASSWORD_DETECTOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            json_str = response["message"]["content"]
            chunk_result = _normalize_result(json.loads(_clean_json_response(json_str)))
            if chunk_result["contains_password"]:
                any_passwords = True
                for key, value in chunk_result["passwords"].items():
                    unique_key = _deduplicate_key(key, value, detected_passwords)
                    detected_passwords[unique_key] = value
        except Exception as e:
            logger.warning(
                "Failed to detect passwords for chunk %d/%d of %s: %s",
                i + 1,
                total_chunks,
                source_display_name,
                e,
            )
            continue

    return {
        "contains_password": any_passwords,
        "passwords": detected_passwords,
    }


def _normalize_estate_response(
    raw_result: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Sanitize model output for estate analysis."""

    if not isinstance(raw_result, dict):
        return {}

    normalized: dict[str, list[dict[str, Any]]] = {}

    for key, value in raw_result.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, list):
            continue
        cleaned_entries: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                cleaned_entries.append(
                    {
                        str(entry_key): entry_value
                        for entry_key, entry_value in item.items()
                        if isinstance(entry_key, str)
                    }
                )
            elif isinstance(item, str):
                cleaned_entries.append({"details": item})
        if cleaned_entries:
            normalized[key] = cleaned_entries

    return normalized


def _merge_estate_results(
    chunked_results: list[dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    """Combine estate analysis results across chunks."""

    merged: dict[str, list[dict[str, Any]]] = {}

    for result in chunked_results:
        for category, entries in result.items():
            if not isinstance(category, str) or not isinstance(entries, list):
                continue
            bucket = merged.setdefault(category, [])
            for entry in entries:
                if isinstance(entry, dict) and entry not in bucket:
                    bucket.append(entry)

    return merged


def analyze_estate_relevant_information(
    text: str,
    *,
    source_name: str | None = None,
    should_abort: AbortCallback | None = None,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Identify estate management details within text."""

    if not text.strip():
        return DEFAULT_ESTATE_RESULT

    text_bytes = len(text.encode("utf-8"))
    source_display_name = _resolve_source_name(source_name)
    logger.debug(
        "Starting estate analysis, text length: %d bytes for %s",
        text_bytes,
        source_display_name,
    )

    instructions = (
        "You are an assistant helping loved ones settle a deceased person's affairs. "
        "Review the supplied text and capture only information that would help an "
        "executor or family member locate important assets, instructions, or "
        "accounts.\n\n"
        "Return a JSON object. Use only these top-level keys when relevant: "
        f"{', '.join(ESTATE_CATEGORY_KEYS)}.\n"
        "Each present key must map to an array of objects. For each item include:\n"
        '  - "item": a concise label for the information (e.g., "Living Will", '
        '"Chase savings account").\n'
        '  - "why_it_matters": a short phrase on why the detail helps wrap up '
        "affairs.\n"
        '  - "details": the critical facts pulled from the text such as account '
        "numbers, "
        "custodian names, login hints, storage locations, or instructions.\n"
        'Add optional keys like "location", "contact", or "reference" when the '
        "text supplies them. Do not fabricate information and omit categories "
        "that have "
        "no findings. If nothing is relevant return an empty JSON object {}.\n"
        "Respond only with raw JSON (no code fences)."
    )

    def _call_model(payload: str) -> dict[str, list[dict[str, Any]]]:
        _maybe_abort(should_abort)
        response = ollama.chat(
            model=TEXT_ANALYZER_MODEL,
            messages=[{"role": "user", "content": payload}],
        )
        json_str = response["message"]["content"]
        parsed = json.loads(_clean_json_response(json_str))
        return _normalize_estate_response(parsed)

    if text_bytes <= 3000:
        prompt = f"{instructions}\n\nText:\n{text}"
        try:
            logger.debug(
                "Sending estate analysis request (single chunk) for %s",
                source_display_name,
            )
            normalized = _call_model(prompt)
            has_info = bool(normalized)
            return {
                "has_estate_relevant_info": has_info,
                "estate_information": normalized,
            }
        except Exception as e:
            logger.warning(
                "Estate analysis failed for %s with Ollama: %s",
                source_display_name,
                e,
            )
            return DEFAULT_ESTATE_RESULT

    chunks = chunk_text(text, max_tokens=2048)
    chunks = _limit_chunk_list(
        chunks,
        max_chunks,
        analysis_name="estate analysis",
        source_name=source_display_name,
    )
    total_chunks = len(chunks)
    chunk_results: list[dict[str, list[dict[str, Any]]]] = []

    for index, chunk in enumerate(chunks, start=1):
        _maybe_abort(should_abort)
        prompt = (
            f"{instructions}\n\n"
            f"This is chunk {index} of {total_chunks} from "
            f"{source_display_name}:\n{chunk}"
        )
        try:
            logger.debug(
                "Sending estate analysis request for chunk %d/%d of %s",
                index,
                total_chunks,
                source_display_name,
            )
            normalized = _call_model(prompt)
            if normalized:
                chunk_results.append(normalized)
        except Exception as e:
            logger.warning(
                "Estate analysis failed for chunk %d/%d of %s: %s",
                index,
                total_chunks,
                source_display_name,
                e,
            )
            continue

    merged = _merge_estate_results(chunk_results)
    has_info = bool(merged)
    logger.debug(
        "Completed estate analysis for %s with %d chunk result(s)",
        source_display_name,
        len(chunk_results),
    )
    return {
        "has_estate_relevant_info": has_info,
        "estate_information": merged,
    }


def analyze_financial_document(
    text: str,
    *,
    source_name: str | None = None,
    should_abort: AbortCallback | None = None,
    max_chunks: int | None = None,
) -> dict[str, Any]:
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
    source_display_name = _resolve_source_name(source_name)
    logger.debug(
        "Starting financial document analysis, text length: %d bytes",
        text_bytes,
    )
    # Check if text needs chunking (over 3000 bytes)
    if text_bytes <= 3000:
        _maybe_abort(should_abort)
        logger.info(
            "Financial analysis processing single chunk for %s", source_display_name
        )
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
            _maybe_abort(should_abort)
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
    chunks = _limit_chunk_list(
        chunks,
        max_chunks,
        analysis_name="financial analysis",
        source_name=source_display_name,
    )
    total_chunks = len(chunks)
    logger.info(
        "Financial analysis processing %d chunk(s) for %s",
        total_chunks,
        source_display_name,
    )
    all_summaries = []
    all_red_flags = set()
    all_incriminating = set()
    confidence_scores = []

    for i, chunk in enumerate(chunks):
        _maybe_abort(should_abort)
        remaining_chunks = total_chunks - (i + 1)
        logger.info(
            "Financial analysis chunk %d/%d for %s (%d remaining)",
            i + 1,
            total_chunks,
            source_display_name,
            remaining_chunks,
        )
        prompt = (
            f"You are a meticulous forensic accountant. Analyze chunk {i+1} of "
            f"{total_chunks} of the following financial document text and provide a "
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
                "Sending Ollama request for financial chunk %d/%d for %s, "
                "prompt length: %d",
                i + 1,
                total_chunks,
                source_display_name,
                len(prompt),
            )
            _maybe_abort(should_abort)
            response = ollama.chat(
                model=CODE_ANALYZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"raw": True},
            )
            logger.debug(
                "Received Ollama response for financial chunk %d/%d for %s",
                i + 1,
                total_chunks,
                source_display_name,
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
                "Failed to analyze financial chunk %d for %s with Ollama: %s",
                i + 1,
                source_display_name,
                e,
            )
            continue

    # Combine results
    if all_summaries:
        _maybe_abort(should_abort)
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
            _maybe_abort(should_abort)
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


def describe_image(
    image_path: str, *, should_abort: AbortCallback | None = None
) -> str:
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
            _maybe_abort(should_abort)
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
        _maybe_abort(should_abort)
        image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode("utf-8")

    try:
        logger.debug(
            "Sending Ollama request for image description, image size: %d bytes",
            len(image_b64),
        )
        _maybe_abort(should_abort)
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


def summarize_video_frames(
    frame_descriptions: list[str], *, should_abort: AbortCallback | None = None
) -> str:
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

    _maybe_abort(should_abort)

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
        _maybe_abort(should_abort)
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
