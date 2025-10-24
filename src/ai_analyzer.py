"""AI-powered analysis utilities for file content."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import httpx
import ollama
from PIL import Image, UnidentifiedImageError

from src.config_utils import (
    build_ollama_options,
    compute_chunk_size,
    get_model_config,
    load_config,
)
from src.text_utils import chunk_text, count_tokens

logger = logging.getLogger(__name__)

# Load config
config = load_config()
models_config = config.get("models", {})
text_analyzer_config = get_model_config(models_config, "text_analyzer")
code_analyzer_config = get_model_config(models_config, "code_analyzer")
image_describer_config = get_model_config(models_config, "image_describer")

TEXT_ANALYZER_MODEL = text_analyzer_config["name"]
TEXT_ANALYZER_CONTEXT_WINDOW = text_analyzer_config.get("context_window")
TEXT_ANALYZER_OPTIONS = build_ollama_options(text_analyzer_config)
CODE_ANALYZER_MODEL = code_analyzer_config["name"]
CODE_ANALYZER_CONTEXT_WINDOW = code_analyzer_config.get("context_window")
CODE_ANALYZER_OPTIONS = build_ollama_options(code_analyzer_config)
IMAGE_DESCRIBER_MODEL = image_describer_config["name"]
IMAGE_DESCRIBER_CONTEXT_WINDOW = image_describer_config.get("context_window")
IMAGE_DESCRIBER_OPTIONS = build_ollama_options(image_describer_config)

password_detector_entry = models_config.get("password_detector")
if password_detector_entry is None:
    password_detector_config = text_analyzer_config
else:
    password_detector_config = get_model_config(models_config, "password_detector")

PASSWORD_DETECTOR_MODEL = password_detector_config["name"]
PASSWORD_DETECTOR_CONTEXT_WINDOW = password_detector_config.get("context_window")
PASSWORD_DETECTOR_OPTIONS = build_ollama_options(password_detector_config)
PROMPT_RESERVE_SAFETY_TOKENS = 128
JSON_RESPONSE_FORMAT = "json"
JSON_OUTPUT_OPTIONS = {"temperature": 0}
STRICT_JSON_REMINDER = (
    "REMINDER: Reply with strictly valid JSON. Use double quotes for all keys and "
    "string values, include commas between fields, and do not add commentary."
)
JSON_SYSTEM_MESSAGE = (
    "You are a JSON generation engine. Every reply MUST be a single valid JSON "
    "object that strictly follows the caller's schema. Do not add explanations, "
    "code fences, or any text outside the JSON object. Use double quotes for all "
    "keys and string values and ensure the JSON is syntactically correct."
)
TOKEN_MARGIN_FACTOR = 2.0
PASSWORD_DETECTOR_MAX_JSON_FAILURES = 5
LLM_DEFAULT_TIMEOUT = 60.0


class LLMTimeoutError(RuntimeError):
    """Raised when an LLM request exceeds the configured timeout."""


def _build_text_single_prompt(text: str) -> str:
    return (
        "You are a document analyst. Analyze the following text and provide a "
        "JSON response with exactly two keys:\n\n"
        '- "summary": a concise paragraph summarizing the main points of the '
        "text.\n"
        '- "mentioned_people": a list of names of people mentioned in the text. '
        "These should be actual names, like 'John Smith' or 'Maria Garcia', "
        "not usernames like 'user123'.\n\n"
        "Example response:\n"
        "{\n"
        '  "summary": "Concise paragraph here.",\n'
        '  "mentioned_people": ["Alice Johnson", "Robert Smith"]\n'
        "}\n\n"
        f"Text: {text}\n\n"
        "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
        "backticks. Return only the raw JSON object."
    )


def _build_text_chunk_prompt(*, chunk: str, index: int, chunk_count: int) -> str:
    return (
        f"You are a document analyst. Analyze chunk {index}/{chunk_count} of the "
        "following text and provide a JSON response with exactly two keys:\n\n"
        '- "summary": a concise paragraph summarizing the main points of this '
        "chunk.\n"
        '- "mentioned_people": a list of names of people mentioned in this '
        "chunk. These should be actual names, like 'John Smith' or "
        "'Maria Garcia', not usernames like 'user123'.\n\n"
        "Example response:\n"
        "{\n"
        '  "summary": "Concise paragraph for this chunk.",\n'
        '  "mentioned_people": ["Alice Johnson"]\n'
        "}\n\n"
        f"Text chunk: {chunk}\n\n"
        "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
        "backticks. Return only the raw JSON object."
    )


PASSWORD_DETECTOR_PROMPT_TEMPLATE = (
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
    "Example response:\n"
    "{\n"
    '  "contains_password": true,\n'
    '  "passwords": {\n'
    '    "email_account": "S3cretPass!"\n'
    "  }\n"
    "}\n\n"
    "Only consider information present in the text. Do not invent entries. "
    "Respond with raw JSON only."
)


def _build_password_single_prompt(text: str) -> str:
    return f"{PASSWORD_DETECTOR_PROMPT_TEMPLATE}\n\nText:\n{text}"


def _build_password_chunk_prompt(*, chunk: str, index: int, chunk_count: int) -> str:
    return (
        f"{PASSWORD_DETECTOR_PROMPT_TEMPLATE}\n\n"
        f"This is chunk {index} of {chunk_count}:\n{chunk}"
    )


ESTATE_CATEGORY_KEYS = [
    "Legal",
    "Financial",
    "Insurance",
    "Digital",
    "Medical",
    "Personal",
]


ESTATE_PLACEHOLDER_VALUES = {
    "",
    "details",
    "detail",
    "unknown",
    "n/a",
    "na",
    "none",
    "placeholder",
}

ESTATE_MIN_FIELDS_PER_ENTRY = 2  # Require "item" plus at least one supporting field.
ESTATE_SKIP_BARE_STRING_LOG = (
    "Skipping estate entry emitted as bare string; expected structured object."
)


ESTATE_ANALYSIS_INSTRUCTIONS = (
    "You are an assistant triaging documents to help loved ones settle a "
    "deceased person's estate. Review the supplied text and record details "
    "only when the passage explicitly mentions something an executor could act "
    "on: legal directives (wills, trusts, POA), financial or insurance "
    "accounts, titled property, debts to resolve, medical directives, digital "
    "logins, or instructions on where to find records.\n\n"
    "Ignore personal narratives, biographies, hobbies, memberships, awards, "
    "and generic life history unless the text clearly ties them to a legal, "
    "financial, or administrative obligation. Do not infer assets—cite only "
    "information that is stated or quoted from the text. Never copy the "
    "example output or invent placeholders.\n\n"
    "Return a JSON object. Use only these top-level keys when relevant: "
    f"{', '.join(ESTATE_CATEGORY_KEYS)}.\n"
    "Each present key must map to an array of objects. For every entry include:\n"
    '  - "item": a concise label (e.g., "Living Will", "Chase savings account").\n'
    '  - "why_it_matters": how this helps settle the estate.\n'
    '  - "details": the exact facts or faithful paraphrase from the text such as '
    "institution names, account numbers, storage locations, instructions, or "
    "contact details.\n"
    'Add optional keys like "location", "contact", or "reference" when the text '
    "supplies them. Skip any entry if you cannot provide meaningful text for "
    '"item", "why_it_matters", and "details"; never output filler strings like '
    '"details", "unknown", "N/A", or "none". If the passage lacks actionable '
    "estate information, return an empty JSON object {}.\n"
    "Respond only with raw JSON (no code fences)."
)


def _build_estate_single_prompt(text: str) -> str:
    return (
        f"{ESTATE_ANALYSIS_INSTRUCTIONS}\n\n"
        "Example response:\n"
        "{\n"
        '  "Legal": [\n'
        "    {\n"
        '      "item": "Last will and testament",\n'
        '      "why_it_matters": "Names executor and divides property",\n'
        '      "details": "Stored in the safe at 123 Main St., combination 1965"\n'
        "    }\n"
        "  ],\n"
        '  "Financial": [\n'
        "    {\n"
        '      "item": "Chase savings account",\n'
        '      "why_it_matters": "Funds available for estate expenses",\n'
        '      "details": "Account ending 1234, branch on 5th Ave"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Text:\n{text}"
    )


def _build_estate_chunk_prompt(
    *, chunk: str, index: int, chunk_count: int, source_name: str
) -> str:
    return (
        f"{ESTATE_ANALYSIS_INSTRUCTIONS}\n\n"
        "Example response:\n"
        "{\n"
        '  "Legal": [\n'
        "    {\n"
        '      "item": "Last will and testament",\n'
        '      "why_it_matters": "Names executor and divides property",\n'
        '      "details": "Stored in the safe at 123 Main St., combination 1965"\n'
        "    }\n"
        "  ],\n"
        '  "Financial": [\n'
        "    {\n"
        '      "item": "Chase savings account",\n'
        '      "why_it_matters": "Funds available for estate expenses",\n'
        '      "details": "Account ending 1234, branch on 5th Ave"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"This is chunk {index} of {chunk_count} from {source_name}:\n{chunk}"
    )


FINANCIAL_ANALYSIS_PROMPT_TEMPLATE = (
    "You are a meticulous forensic accountant. Analyze the following "
    "financial document text and provide a JSON response with exactly "
    "four keys:\n\n"
    '- "summary": a concise paragraph summarizing the document.\n'
    '- "potential_red_flags": a list of potential red flags or irregularities.\n'
    '- "incriminating_items": a list of items that could be incriminating.\n'
    '- "confidence_score": a numerical score from 0 to 100 indicating '
    "confidence in the analysis.\n\n"
    "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
    "backticks. Return only the raw JSON object."
)


def _build_financial_single_prompt(text: str) -> str:
    return (
        f"{FINANCIAL_ANALYSIS_PROMPT_TEMPLATE}\n\n"
        "Example response:\n"
        "{\n"
        '  "summary": "Overall summary of the document.",\n'
        '  "potential_red_flags": ["Late payment noted"],\n'
        '  "incriminating_items": ["Unreported cash deposit"],\n'
        '  "confidence_score": 78\n'
        "}\n\n"
        f"Text:\n{text}"
    )


def _build_financial_chunk_prompt(*, chunk: str, index: int, chunk_count: int) -> str:
    return (
        f"You are a meticulous forensic accountant. "
        f"Analyze chunk {index}/{chunk_count} of the following financial document "
        "text and provide a JSON response with exactly four keys:\n\n"
        '- "summary": a concise paragraph summarizing this chunk.\n'
        '- "potential_red_flags": a list of potential red flags or irregularities '
        "in this chunk.\n"
        '- "incriminating_items": a list of items that could be incriminating in '
        "this chunk.\n"
        '- "confidence_score": a numerical score from 0 to 100 indicating '
        "confidence in the analysis of this chunk.\n\n"
        "Example response:\n"
        "{\n"
        '  "summary": "Chunk-specific summary.",\n'
        '  "potential_red_flags": [],\n'
        '  "incriminating_items": [],\n'
        '  "confidence_score": 60\n'
        "}\n\n"
        f"Text chunk: {chunk}\n\n"
        "Respond only with valid JSON. Do not wrap the JSON in code blocks or "
        "backticks. Return only the raw JSON object."
    )


def _prepare_chunks(
    text: str,
    *,
    initial_limit: int,
    context_window: int | None,
    prompt_factory: Callable[[str, int, int], str],
    minimum_limit: int = 128,
) -> tuple[list[str], int]:
    """Chunk text while ensuring prompts fit within the model context window."""

    limit = max(int(initial_limit / TOKEN_MARGIN_FACTOR), minimum_limit)

    while True:
        chunks = chunk_text(text, max_tokens=limit)
        if not chunks:
            return [], limit

        if context_window is None:
            return chunks, limit

        chunk_count = len(chunks)
        fits = True
        for idx, chunk in enumerate(chunks, start=1):
            prompt = prompt_factory(chunk, idx, chunk_count)
            prompt_tokens = count_tokens(prompt)
            adjusted_tokens = int(prompt_tokens * TOKEN_MARGIN_FACTOR)
            if prompt_tokens > context_window:
                fits = False
                break
            if adjusted_tokens > context_window:
                fits = False
                break

        if fits:
            return chunks, limit

        new_limit = max(limit // 2, minimum_limit)
        logger.debug(
            "Prompt exceeded context window (limit=%d, new_limit=%d, context=%s)",
            limit,
            new_limit,
            context_window,
        )
        if new_limit == limit:
            if limit == minimum_limit:
                return chunks, limit
            limit = minimum_limit
        else:
            limit = new_limit


_TEXT_CHUNK_PROMPT_BASE_TOKENS = count_tokens(
    _build_text_chunk_prompt(chunk="", index=1, chunk_count=1)
)
_ESTATE_CHUNK_PROMPT_BASE_TOKENS = count_tokens(
    _build_estate_chunk_prompt(
        chunk="",
        index=1,
        chunk_count=1,
        source_name="example.txt",
    )
)
TEXT_ANALYZER_PROMPT_RESERVE = (
    max(_TEXT_CHUNK_PROMPT_BASE_TOKENS, _ESTATE_CHUNK_PROMPT_BASE_TOKENS)
    + PROMPT_RESERVE_SAFETY_TOKENS
)
TEXT_ANALYZER_CHUNK_TOKENS = compute_chunk_size(
    TEXT_ANALYZER_CONTEXT_WINDOW,
    reserve_tokens=TEXT_ANALYZER_PROMPT_RESERVE,
)

_PASSWORD_CHUNK_PROMPT_BASE_TOKENS = count_tokens(
    _build_password_chunk_prompt(chunk="", index=1, chunk_count=1)
)
PASSWORD_DETECTOR_PROMPT_RESERVE = (
    _PASSWORD_CHUNK_PROMPT_BASE_TOKENS + PROMPT_RESERVE_SAFETY_TOKENS
)
PASSWORD_DETECTOR_CHUNK_TOKENS = compute_chunk_size(
    PASSWORD_DETECTOR_CONTEXT_WINDOW,
    reserve_tokens=PASSWORD_DETECTOR_PROMPT_RESERVE,
)

_FINANCIAL_CHUNK_PROMPT_BASE_TOKENS = count_tokens(
    _build_financial_chunk_prompt(chunk="", index=1, chunk_count=1)
)
CODE_ANALYZER_PROMPT_RESERVE = (
    _FINANCIAL_CHUNK_PROMPT_BASE_TOKENS + PROMPT_RESERVE_SAFETY_TOKENS
)
CODE_ANALYZER_CHUNK_TOKENS = compute_chunk_size(
    CODE_ANALYZER_CONTEXT_WINDOW,
    reserve_tokens=CODE_ANALYZER_PROMPT_RESERVE,
)
logger.debug(
    "Model configuration: text=%s ctx=%s chunk=%s; "
    "code=%s ctx=%s chunk=%s; image=%s ctx=%s; password=%s ctx=%s chunk=%s",
    TEXT_ANALYZER_MODEL,
    TEXT_ANALYZER_CONTEXT_WINDOW,
    TEXT_ANALYZER_CHUNK_TOKENS,
    CODE_ANALYZER_MODEL,
    CODE_ANALYZER_CONTEXT_WINDOW,
    CODE_ANALYZER_CHUNK_TOKENS,
    IMAGE_DESCRIBER_MODEL,
    IMAGE_DESCRIBER_CONTEXT_WINDOW,
    PASSWORD_DETECTOR_MODEL,
    PASSWORD_DETECTOR_CONTEXT_WINDOW,
    PASSWORD_DETECTOR_CHUNK_TOKENS,
)

DEFAULT_PASSWORD_RESULT = {
    "contains_password": False,
    "passwords": {},
    "_chunk_count": 0,
}
DEFAULT_ESTATE_RESULT = {
    "has_estate_relevant_info": False,
    "estate_information": {},
    "_chunk_count": 0,
}


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


def _extract_json_segment(text: str) -> str | None:
    """Attempt to extract the first valid JSON object or array from text."""

    opening_chars = {"{": "}", "[": "]"}
    start_index = None
    start_char = None

    for candidate, closing in opening_chars.items():
        idx = text.find(candidate)
        if idx != -1 and (start_index is None or idx < start_index):
            start_index = idx
            start_char = candidate

    if start_index is None or start_char is None:
        return None

    stack: list[str] = []
    in_string = False
    escape = False

    for position in range(start_index, len(text)):
        char = text[position]

        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char in opening_chars:
            stack.append(opening_chars[char])
        elif char in opening_chars.values():
            if not stack:
                return None
            expected = stack.pop()
            if char != expected:
                continue
            if not stack:
                return text[start_index : position + 1]

    return None


def _parse_json_payload(payload: str) -> Any:
    cleaned = _clean_json_response(payload)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _extract_json_segment(cleaned)
        if candidate and candidate != cleaned:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                logger.debug("Failed to parse extracted JSON segment: %s", candidate)
        raise


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


def _ollama_chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    options: dict[str, Any] | None = None,
    response_format: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Invoke Ollama chat with optional configuration."""
    request: dict[str, Any] = {"model": model, "messages": messages}
    if options:
        request["options"] = dict(options)
    if response_format:
        request["format"] = response_format
    if timeout and timeout > 0:
        client = ollama.Client(timeout=timeout)
        return client.chat(**request)
    return ollama.chat(**request)


def _combine_options(*option_dicts: dict[str, Any] | None) -> dict[str, Any]:
    """Merge multiple Ollama option dictionaries."""
    merged: dict[str, Any] = {}
    for opts in option_dicts:
        if not opts:
            continue
        merged.update(opts)
    return merged


def _compute_dynamic_timeout(history: list[float]) -> tuple[float, float]:
    """Return (baseline_p75, timeout_limit)."""

    valid = [duration for duration in history if duration > 0]
    if len(valid) < 4:
        baseline = LLM_DEFAULT_TIMEOUT
    else:
        valid.sort()
        percentile_index = math.ceil(0.75 * (len(valid) + 1)) - 1
        percentile_index = min(max(percentile_index, 0), len(valid) - 1)
        baseline_candidate = valid[percentile_index]
        baseline = baseline_candidate if baseline_candidate > 0 else LLM_DEFAULT_TIMEOUT
    timeout_limit = max(baseline * 5, LLM_DEFAULT_TIMEOUT)
    return baseline, timeout_limit


def _run_with_timeout(func: Callable[[], Any], timeout: float | None) -> Any:
    if not timeout or timeout <= 0:
        return func()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:  # pragma: no cover - timing
            future.cancel()
            raise LLMTimeoutError("LLM request timed out") from exc


def _request_json_response(
    *,
    model: str,
    prompt: str,
    options: dict[str, Any] | None,
    should_abort: AbortCallback | None,
    context: str,
    max_attempts: int = 2,
    max_duration: float | None = None,
) -> Any:
    """Send a chat request expecting JSON and retry with stricter instructions."""

    base_prompt = prompt
    prompt_to_send = prompt

    timeout_budget = max(max_duration or 0, LLM_DEFAULT_TIMEOUT)

    for attempt in range(1, max_attempts + 1):
        _maybe_abort(should_abort)

        def _do_request() -> dict[str, Any]:
            try:
                return _ollama_chat(
                    model,
                    [
                        {"role": "system", "content": JSON_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt_to_send},
                    ],
                    options=options,
                    response_format=JSON_RESPONSE_FORMAT,
                    timeout=timeout_budget,
                )
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError("LLM request timed out") from exc

        try:
            response = _run_with_timeout(_do_request, timeout_budget)
        except LLMTimeoutError:
            logger.warning(
                "%s timed out after %.2f seconds (attempt %d/%d)",
                context,
                timeout_budget,
                attempt,
                max_attempts,
            )
            raise

        content = response["message"]["content"]

        try:
            return _parse_json_payload(content)
        except json.JSONDecodeError as exc:
            logger.debug(
                "Model response failed JSON decode for %s (attempt %d/%d): %s",
                context,
                attempt,
                max_attempts,
                content,
            )
            if attempt >= max_attempts:
                raise
            logger.warning(
                "%s JSON decode failed (attempt %d/%d): %s; "
                "retrying with strict instructions",
                context,
                attempt,
                max_attempts,
                exc,
            )
            prompt_to_send = f"{base_prompt}\n\n{STRICT_JSON_REMINDER}"
            continue


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
    if not text.strip():
        return {
            "summary": "",
            "mentioned_people": [],
            "_chunk_count": 0,
        }

    text_bytes = len(text.encode("utf-8"))
    source_display_name = _resolve_source_name(source_name)
    logger.debug(
        "Starting text content analysis, text length: %d bytes",
        text_bytes,
    )
    chunk_token_limit = TEXT_ANALYZER_CHUNK_TOKENS
    if text_bytes <= 3000:
        _maybe_abort(should_abort)
        logger.info("Text analysis processing single chunk for %s", source_display_name)
        # Single chunk processing
        prompt = _build_text_single_prompt(text)

        prompt_token_estimate = int(count_tokens(prompt) * TOKEN_MARGIN_FACTOR)
        if (
            TEXT_ANALYZER_CONTEXT_WINDOW
            and prompt_token_estimate > TEXT_ANALYZER_CONTEXT_WINDOW
        ):
            logger.debug(
                "Single chunk prompt exceeds context for %s "
                "(estimated tokens=%d, context=%s); switching to chunked mode",
                source_display_name,
                prompt_token_estimate,
                TEXT_ANALYZER_CONTEXT_WINDOW,
            )
            chunk_token_limit = max(
                int(TEXT_ANALYZER_CHUNK_TOKENS / TOKEN_MARGIN_FACTOR), 128
            )
        else:
            try:
                logger.debug(
                    "Sending Ollama text analysis request (single chunk), "
                    "prompt length: %d",
                    len(prompt),
                )
                result = _request_json_response(
                    model=TEXT_ANALYZER_MODEL,
                    prompt=prompt,
                    options=_combine_options(
                        TEXT_ANALYZER_OPTIONS, JSON_OUTPUT_OPTIONS
                    ),
                    should_abort=should_abort,
                    context=f"text analysis single chunk for {source_display_name}",
                )
                if isinstance(result, dict):
                    result["_chunk_count"] = 1
                return result
            except json.JSONDecodeError as decode_error:
                logger.warning(
                    "Single-chunk text analysis JSON decode failed for %s: %s; "
                    "retrying with chunks",
                    source_display_name,
                    decode_error,
                )
                chunk_token_limit = max(
                    int(TEXT_ANALYZER_CHUNK_TOKENS / TOKEN_MARGIN_FACTOR), 128
                )
            except Exception as e:
                logger.warning("Failed to analyze text content with Ollama: %s", e)
                return {
                    "summary": "Analysis unavailable - Ollama not accessible",
                    "mentioned_people": [],
                    "_chunk_count": 0,
                }

    # Multi-chunk processing
    chunks, chunk_token_limit = _prepare_chunks(
        text,
        initial_limit=chunk_token_limit,
        context_window=TEXT_ANALYZER_CONTEXT_WINDOW,
        prompt_factory=lambda chunk, idx, total: _build_text_chunk_prompt(
            chunk=chunk, index=idx, chunk_count=total
        ),
    )
    chunks = _limit_chunk_list(
        chunks,
        max_chunks,
        analysis_name="text analysis",
        source_name=source_display_name,
    )
    chunk_count = len(chunks)
    logger.info(
        "Text analysis processing %d chunk(s) for %s (chunk_token_limit=%d)",
        chunk_count,
        source_display_name,
        chunk_token_limit,
    )
    logger.info(
        "Text analysis processing %d chunk(s) for %s",
        chunk_count,
        source_display_name,
    )
    all_summaries = []
    all_people = set()
    chunk_durations: list[float] = []

    for i, chunk in enumerate(chunks):
        _maybe_abort(should_abort)
        remaining_chunks = chunk_count - (i + 1)
        logger.info(
            "Text analysis chunk %d/%d for %s (%d remaining)",
            i + 1,
            chunk_count,
            source_display_name,
            remaining_chunks,
        )
        prompt = _build_text_chunk_prompt(
            chunk=chunk,
            index=i + 1,
            chunk_count=chunk_count,
        )
        _baseline, timeout_limit = _compute_dynamic_timeout(chunk_durations)
        start_time = time.monotonic()

        try:
            logger.debug(
                "Sending Ollama request for text chunk %d/%d for %s, prompt length: %d",
                i + 1,
                chunk_count,
                source_display_name,
                len(prompt),
            )
            chunk_result = _request_json_response(
                model=TEXT_ANALYZER_MODEL,
                prompt=prompt,
                options=_combine_options(TEXT_ANALYZER_OPTIONS, JSON_OUTPUT_OPTIONS),
                should_abort=should_abort,
                context=(f"text chunk {i + 1}/{chunk_count} for {source_display_name}"),
                max_duration=timeout_limit,
            )
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            all_summaries.append(chunk_result.get("summary", ""))
            all_people.update(chunk_result.get("mentioned_people", []))
        except LLMTimeoutError:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration if duration > 0 else (timeout_limit or 0))
            logger.warning(
                "Text analysis chunk %d/%d for %s timed out after %.2fs; skipping",
                i + 1,
                chunk_count,
                source_display_name,
                timeout_limit or 0.0,
            )
            continue
        except json.JSONDecodeError as exc:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            logger.warning(
                "Text analysis chunk %d/%d JSON decode failed for %s after retries: %s",
                i + 1,
                chunk_count,
                source_display_name,
                exc,
            )
            continue
        except Exception as e:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
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
            response = _ollama_chat(
                TEXT_ANALYZER_MODEL,
                [{"role": "user", "content": combined_summary_prompt}],
                options=TEXT_ANALYZER_OPTIONS,
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
        "_chunk_count": chunk_count,
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

    if text_bytes <= 3000:
        prompt = _build_password_single_prompt(text)
        prompt_token_estimate = int(count_tokens(prompt) * TOKEN_MARGIN_FACTOR)
        if (
            PASSWORD_DETECTOR_CONTEXT_WINDOW
            and prompt_token_estimate > PASSWORD_DETECTOR_CONTEXT_WINDOW
        ):
            logger.debug(
                "Single chunk password prompt exceeds context for %s "
                "(estimated tokens=%d, context=%s); switching to chunked mode",
                source_display_name,
                prompt_token_estimate,
                PASSWORD_DETECTOR_CONTEXT_WINDOW,
            )
        else:
            try:
                logger.debug(
                    "Sending Ollama password detection request (single chunk) for %s",
                    source_display_name,
                )
                raw_result = _request_json_response(
                    model=PASSWORD_DETECTOR_MODEL,
                    prompt=prompt,
                    options=_combine_options(
                        PASSWORD_DETECTOR_OPTIONS, JSON_OUTPUT_OPTIONS
                    ),
                    should_abort=should_abort,
                    context=(
                        f"password detection single chunk for {source_display_name}"
                    ),
                )
                result = _normalize_result(raw_result)
                result["_chunk_count"] = 1
                return result
            except json.JSONDecodeError as exc:
                logger.warning(
                    (
                        "Password detection JSON decode failed for %s: %s; "
                        "switching to chunks"
                    ),
                    source_display_name,
                    exc,
                )
            except Exception as e:
                logger.warning(
                    "Failed to detect passwords in %s with Ollama: %s",
                    source_display_name,
                    e,
                )
                return DEFAULT_PASSWORD_RESULT

    chunks, _ = _prepare_chunks(
        text,
        initial_limit=PASSWORD_DETECTOR_CHUNK_TOKENS,
        context_window=PASSWORD_DETECTOR_CONTEXT_WINDOW,
        prompt_factory=lambda chunk, idx, total: _build_password_chunk_prompt(
            chunk=chunk, index=idx, chunk_count=total
        ),
    )
    chunks = _limit_chunk_list(
        chunks,
        max_chunks,
        analysis_name="password detection",
        source_name=source_display_name,
    )
    chunk_count = len(chunks)
    detected_passwords: dict[str, str] = {}
    any_passwords = False
    json_failure_streak = 0
    chunk_durations: list[float] = []
    fallback_logged = False
    if chunk_count:
        logger.info(
            "Password detection processing %d chunk(s) for %s",
            chunk_count,
            source_display_name,
        )

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
        remaining_chunks = chunk_count - (i + 1)
        logger.info(
            "Password detection chunk %d/%d for %s (%d remaining)",
            i + 1,
            chunk_count,
            source_display_name,
            remaining_chunks,
        )
        prompt = _build_password_chunk_prompt(
            chunk=chunk,
            index=i + 1,
            chunk_count=chunk_count,
        )
        _baseline, timeout_limit = _compute_dynamic_timeout(chunk_durations)
        use_fallback_only = json_failure_streak >= PASSWORD_DETECTOR_MAX_JSON_FAILURES
        if use_fallback_only:
            if not fallback_logged:
                logger.info(
                    (
                        "Password detection fallback engaged after %d consecutive JSON "
                        "failures"
                    ),
                    json_failure_streak,
                )
                fallback_logged = True
            fallback_passwords = _fallback_detect_secrets(chunk)
            if fallback_passwords:
                any_passwords = True
                for key, value in fallback_passwords.items():
                    unique_key = _deduplicate_key(key, value, detected_passwords)
                    detected_passwords[unique_key] = value
            continue

        start_time = time.monotonic()
        try:
            raw_result = _request_json_response(
                model=PASSWORD_DETECTOR_MODEL,
                prompt=prompt,
                options=_combine_options(
                    PASSWORD_DETECTOR_OPTIONS, JSON_OUTPUT_OPTIONS
                ),
                should_abort=should_abort,
                context=(
                    f"password detection chunk {i + 1}/{chunk_count} for "
                    f"{source_display_name}"
                ),
                max_duration=timeout_limit,
            )
            chunk_result = _normalize_result(raw_result)
            if chunk_result["contains_password"]:
                any_passwords = True
                for key, value in chunk_result["passwords"].items():
                    unique_key = _deduplicate_key(key, value, detected_passwords)
                    detected_passwords[unique_key] = value
            json_failure_streak = 0
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
        except LLMTimeoutError:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration if duration > 0 else (timeout_limit or 0))
            json_failure_streak += 1
            logger.warning(
                "Password detection chunk %d/%d for %s timed out after %.2fs; skipping",
                i + 1,
                chunk_count,
                source_display_name,
                timeout_limit or 0.0,
            )
            fallback_passwords = _fallback_detect_secrets(chunk)
            if fallback_passwords:
                any_passwords = True
                for key, value in fallback_passwords.items():
                    unique_key = _deduplicate_key(key, value, detected_passwords)
                    detected_passwords[unique_key] = value
            continue
        except json.JSONDecodeError as exc:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            logger.warning(
                (
                    "Password detection chunk %d/%d JSON decode failed for %s "
                    "after retries: %s"
                ),
                i + 1,
                chunk_count,
                source_display_name,
                exc,
            )
            json_failure_streak += 1
            fallback_passwords = _fallback_detect_secrets(chunk)
            if fallback_passwords:
                any_passwords = True
                for key, value in fallback_passwords.items():
                    unique_key = _deduplicate_key(key, value, detected_passwords)
                    detected_passwords[unique_key] = value
            continue
        except Exception as e:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            logger.warning(
                "Failed to detect passwords for chunk %d/%d of %s: %s",
                i + 1,
                chunk_count,
                source_display_name,
                e,
            )
            json_failure_streak += 1
            fallback_passwords = _fallback_detect_secrets(chunk)
            if fallback_passwords:
                any_passwords = True
                for key, value in fallback_passwords.items():
                    unique_key = _deduplicate_key(key, value, detected_passwords)
                    detected_passwords[unique_key] = value
            continue

    return {
        "contains_password": any_passwords,
        "passwords": detected_passwords,
        "_chunk_count": chunk_count,
    }


@lru_cache(maxsize=1)
def _load_detect_secrets_plugins():
    try:
        from detect_secrets.plugins.common import initialize
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("detect-secrets not installed; skipping fallback password scan")
        return None

    try:
        return initialize.initialize_plugins()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialize detect-secrets plugins: %s", exc)
        return None


def _fallback_detect_secrets(chunk_text: str) -> dict[str, str]:
    plugins = _load_detect_secrets_plugins()
    if not plugins:
        return {}

    detected: dict[str, str] = {}
    for line_number, line in enumerate(chunk_text.splitlines(), start=1):
        for plugin in plugins:
            try:
                findings = plugin.analyze_line(line, line_number, filename="<chunk>")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "detect-secrets plugin %s failed on line %d: %s",
                    plugin.__class__.__name__,
                    line_number,
                    exc,
                )
                continue
            if not findings:
                continue

            if isinstance(findings, dict):
                candidates = findings.values()
            elif isinstance(findings, (list, tuple, set)):
                candidates = findings
            else:
                candidates = [findings]

            for candidate in candidates:
                secret_value = getattr(candidate, "secret_value", None)
                if callable(secret_value):
                    try:
                        secret_value = secret_value()
                    except Exception:  # pragma: no cover - defensive
                        secret_value = None
                if not secret_value:
                    continue

                base_label = (
                    getattr(plugin, "secret_type", plugin.__class__.__name__)
                    .replace(" ", "_")
                    .lower()
                )
                suffix = 1
                label = f"{base_label}_{line_number}_{suffix}"
                while label in detected and detected[label] != secret_value:
                    suffix += 1
                    label = f"{base_label}_{line_number}_{suffix}"
                detected[label] = secret_value

    if detected:
        logger.info(
            "detect-secrets fallback identified %d candidate(s) in password chunk",
            len(detected),
        )
    return detected


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
            if isinstance(item, str):
                logger.debug(ESTATE_SKIP_BARE_STRING_LOG)
                continue
            if not isinstance(item, dict):
                logger.debug(
                    "Skipping estate entry with unsupported type %s; expected dict.",
                    type(item).__name__,
                )
                continue
            cleaned_entry: dict[str, Any] = {}
            for entry_key, entry_value in item.items():
                if not isinstance(entry_key, str):
                    continue
                if isinstance(entry_value, str):
                    stripped = entry_value.strip()
                    if stripped:
                        cleaned_entry[entry_key] = stripped
                elif entry_value is not None:
                    cleaned_entry[entry_key] = entry_value
            if cleaned_entry:
                item_value = cleaned_entry.get("item")
                if not isinstance(item_value, str):
                    continue
                item_trimmed = item_value.strip()
                if (
                    not item_trimmed
                    or item_trimmed.lower() in ESTATE_PLACEHOLDER_VALUES
                ):
                    continue
                cleaned_entry["item"] = item_trimmed

                # Remove placeholder text from other string fields.
                keys_to_prune: list[str] = []
                for entry_key, entry_value in cleaned_entry.items():
                    if entry_key == "item":
                        continue
                    if isinstance(entry_value, str):
                        trimmed_value = entry_value.strip()
                        if (
                            not trimmed_value
                            or trimmed_value.lower() in ESTATE_PLACEHOLDER_VALUES
                        ):
                            keys_to_prune.append(entry_key)
                        else:
                            cleaned_entry[entry_key] = trimmed_value
                    elif entry_value is None:
                        keys_to_prune.append(entry_key)
                for key_to_remove in keys_to_prune:
                    cleaned_entry.pop(key_to_remove, None)

                # Enforce "item" plus at least one supporting field.
                if len(cleaned_entry) < ESTATE_MIN_FIELDS_PER_ENTRY:
                    continue

                cleaned_entries.append(cleaned_entry)
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
        return {
            **DEFAULT_ESTATE_RESULT,
            "_chunk_count": 0,
        }

    text_bytes = len(text.encode("utf-8"))
    source_display_name = _resolve_source_name(source_name)
    logger.debug(
        "Starting estate analysis, text length: %d bytes for %s",
        text_bytes,
        source_display_name,
    )

    def _call_model(
        payload: str,
        *,
        context: str,
        max_duration: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        parsed = _request_json_response(
            model=TEXT_ANALYZER_MODEL,
            prompt=payload,
            options=_combine_options(TEXT_ANALYZER_OPTIONS, JSON_OUTPUT_OPTIONS),
            should_abort=should_abort,
            context=context,
            max_duration=max_duration,
        )
        return _normalize_estate_response(parsed)

    if text_bytes <= 3000:
        prompt = _build_estate_single_prompt(text)
        prompt_token_estimate = int(count_tokens(prompt) * TOKEN_MARGIN_FACTOR)
        if (
            TEXT_ANALYZER_CONTEXT_WINDOW
            and prompt_token_estimate > TEXT_ANALYZER_CONTEXT_WINDOW
        ):
            logger.debug(
                "Single chunk estate prompt exceeds context for %s "
                "(estimated tokens=%d, context=%s); switching to chunked mode",
                source_display_name,
                prompt_token_estimate,
                TEXT_ANALYZER_CONTEXT_WINDOW,
            )
        else:
            try:
                logger.debug(
                    "Sending estate analysis request (single chunk) for %s",
                    source_display_name,
                )
                normalized = _call_model(
                    prompt,
                    context=f"estate analysis single chunk for {source_display_name}",
                )
                has_info = bool(normalized)
                return {
                    "has_estate_relevant_info": has_info,
                    "estate_information": normalized,
                    "_chunk_count": 1,
                }
            except json.JSONDecodeError as exc:
                logger.warning(
                    (
                        "Estate analysis JSON decode failed for %s: %s; "
                        "retrying with chunked analysis"
                    ),
                    source_display_name,
                    exc,
                )
                # Fall through to chunked processing
            except Exception as e:
                logger.warning(
                    "Estate analysis failed for %s with Ollama: %s",
                    source_display_name,
                    e,
                )
                return {
                    **DEFAULT_ESTATE_RESULT,
                    "_chunk_count": 0,
                }

    chunks, chunk_token_limit = _prepare_chunks(
        text,
        initial_limit=TEXT_ANALYZER_CHUNK_TOKENS,
        context_window=TEXT_ANALYZER_CONTEXT_WINDOW,
        prompt_factory=lambda chunk, idx, total: _build_estate_chunk_prompt(
            chunk=chunk,
            index=idx,
            chunk_count=total,
            source_name=source_display_name,
        ),
    )
    chunks = _limit_chunk_list(
        chunks,
        max_chunks,
        analysis_name="estate analysis",
        source_name=source_display_name,
    )
    chunk_count = len(chunks)
    chunk_results: list[dict[str, list[dict[str, Any]]]] = []
    chunk_durations: list[float] = []
    if chunk_count:
        logger.info(
            "Estate analysis processing %d chunk(s) for %s (chunk_token_limit=%d)",
            chunk_count,
            source_display_name,
            chunk_token_limit,
        )

    for index, chunk in enumerate(chunks, start=1):
        _maybe_abort(should_abort)
        remaining = max(chunk_count - index, 0)
        logger.info(
            "Estate analysis chunk %d/%d for %s (%d remaining)",
            index,
            chunk_count,
            source_display_name,
            remaining,
        )
        prompt = _build_estate_chunk_prompt(
            chunk=chunk,
            index=index,
            chunk_count=chunk_count,
            source_name=source_display_name,
        )
        chunk_label = f"{index}/{chunk_count}"
        _baseline, timeout_limit = _compute_dynamic_timeout(chunk_durations)
        start_time = time.monotonic()
        try:
            logger.debug(
                "Sending estate analysis request for chunk %s of %s",
                chunk_label,
                source_display_name,
            )
            normalized = _call_model(
                prompt,
                context="estate analysis chunk {} for {}".format(
                    chunk_label, source_display_name
                ),
                max_duration=timeout_limit,
            )
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            if normalized:
                chunk_results.append(normalized)
        except LLMTimeoutError:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration if duration > 0 else (timeout_limit or 0))
            logger.warning(
                "Estate analysis chunk %s for %s timed out after %.2fs; skipping",
                chunk_label,
                source_display_name,
                timeout_limit or 0.0,
            )
            continue
        except json.JSONDecodeError as exc:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            logger.warning(
                "Estate analysis chunk %s JSON decode failed for %s after retries: %s",
                chunk_label,
                source_display_name,
                exc,
            )
            continue
        except Exception as e:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            logger.warning(
                "Estate analysis failed for chunk %d/%d of %s: %s",
                index,
                chunk_count,
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
        "_chunk_count": chunk_count,
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
    if not text.strip():
        return {
            "summary": "",
            "potential_red_flags": [],
            "incriminating_items": [],
            "confidence_score": 0,
            "_chunk_count": 0,
        }

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
        prompt = _build_financial_single_prompt(text)
        prompt_token_estimate = int(count_tokens(prompt) * TOKEN_MARGIN_FACTOR)
        if (
            CODE_ANALYZER_CONTEXT_WINDOW
            and prompt_token_estimate > CODE_ANALYZER_CONTEXT_WINDOW
        ):
            logger.debug(
                "Single chunk financial prompt exceeds context for %s "
                "(estimated tokens=%d, context=%s); switching to chunked mode",
                source_display_name,
                prompt_token_estimate,
                CODE_ANALYZER_CONTEXT_WINDOW,
            )
        else:
            try:
                logger.debug(
                    "Sending Ollama financial analysis request (single chunk), "
                    "prompt length: %d",
                    len(prompt),
                )
                result = _request_json_response(
                    model=CODE_ANALYZER_MODEL,
                    prompt=prompt,
                    options=_combine_options(
                        CODE_ANALYZER_OPTIONS, {"raw": True}, JSON_OUTPUT_OPTIONS
                    ),
                    should_abort=should_abort,
                    context=(
                        f"financial analysis single chunk for {source_display_name}"
                    ),
                )
                if isinstance(result, dict):
                    result["_chunk_count"] = 1
                return result
            except json.JSONDecodeError as exc:
                logger.warning(
                    (
                        "Financial analysis JSON decode failed for %s: %s; "
                        "switching to chunks"
                    ),
                    source_display_name,
                    exc,
                )
            except Exception as e:
                logger.warning(
                    "Failed to analyze financial document with Ollama: %s", e
                )
                return {
                    "summary": "Analysis unavailable - Ollama not accessible",
                    "potential_red_flags": [],
                    "incriminating_items": [],
                    "confidence_score": 0,
                    "_chunk_count": 0,
                }

    # Multi-chunk processing
    chunks, _ = _prepare_chunks(
        text,
        initial_limit=CODE_ANALYZER_CHUNK_TOKENS,
        context_window=CODE_ANALYZER_CONTEXT_WINDOW,
        prompt_factory=lambda chunk, idx, total: _build_financial_chunk_prompt(
            chunk=chunk,
            index=idx,
            chunk_count=total,
        ),
    )
    chunks = _limit_chunk_list(
        chunks,
        max_chunks,
        analysis_name="financial analysis",
        source_name=source_display_name,
    )
    chunk_count = len(chunks)
    if chunk_count:
        logger.info(
            "Financial analysis processing %d chunk(s) for %s",
            chunk_count,
            source_display_name,
        )
    all_summaries = []
    all_red_flags = set()
    all_incriminating = set()
    confidence_scores = []
    chunk_durations: list[float] = []

    for i, chunk in enumerate(chunks):
        _maybe_abort(should_abort)
        chunk_index = i + 1
        remaining_chunks = chunk_count - chunk_index
        chunk_label = f"{chunk_index}/{chunk_count}"
        logger.info(
            "Financial analysis chunk %d/%d for %s (%d remaining)",
            chunk_index,
            chunk_count,
            source_display_name,
            remaining_chunks,
        )
        prompt = _build_financial_chunk_prompt(
            chunk=chunk,
            index=i + 1,
            chunk_count=chunk_count,
        )
        _baseline, timeout_limit = _compute_dynamic_timeout(chunk_durations)
        start_time = time.monotonic()

        try:
            logger.debug(
                "Sending financial chunk %s request for %s (prompt len=%d)",
                chunk_label,
                source_display_name,
                len(prompt),
            )
            chunk_result = _request_json_response(
                model=CODE_ANALYZER_MODEL,
                prompt=prompt,
                options=_combine_options(
                    CODE_ANALYZER_OPTIONS, {"raw": True}, JSON_OUTPUT_OPTIONS
                ),
                should_abort=should_abort,
                context=(
                    f"financial analysis chunk {chunk_label} for "
                    f"{source_display_name}"
                ),
                max_duration=timeout_limit,
            )
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            all_summaries.append(chunk_result.get("summary", ""))
            all_red_flags.update(chunk_result.get("potential_red_flags", []))
            all_incriminating.update(chunk_result.get("incriminating_items", []))
            if isinstance(chunk_result.get("confidence_score"), (int, float)):
                confidence_scores.append(chunk_result["confidence_score"])
        except LLMTimeoutError:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration if duration > 0 else (timeout_limit or 0))
            logger.warning(
                "Financial analysis chunk %s for %s timed out after %.2fs; skipping",
                chunk_label,
                source_display_name,
                timeout_limit or 0.0,
            )
            continue
        except json.JSONDecodeError as exc:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
            logger.warning(
                (
                    "Financial analysis chunk %s JSON decode failed for %s "
                    "after retries: %s"
                ),
                chunk_label,
                source_display_name,
                exc,
            )
            continue
        except Exception as e:
            duration = time.monotonic() - start_time
            chunk_durations.append(duration)
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
            response = _ollama_chat(
                CODE_ANALYZER_MODEL,
                [{"role": "user", "content": combined_summary_prompt}],
                options=_combine_options(CODE_ANALYZER_OPTIONS, {"raw": True}),
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
        "_chunk_count": chunk_count,
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
        response = _ollama_chat(
            IMAGE_DESCRIBER_MODEL,
            [
                {
                    "role": "user",
                    "content": "Describe this image in detail.",
                    "images": [image_b64],
                }
            ],
            options=IMAGE_DESCRIBER_OPTIONS,
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
        response = _ollama_chat(
            TEXT_ANALYZER_MODEL,
            [{"role": "user", "content": prompt}],
            options=TEXT_ANALYZER_OPTIONS,
        )
        logger.debug("Received Ollama response for video summarization")
        logger.debug("Completed video frame summarization")
        return response["message"]["content"]
    except Exception as e:
        logger.warning("Failed to summarize video frames with Ollama: %s", e)
        # Fallback: return a simple concatenation
        return "Video summary: " + " ".join(frame_descriptions[:3]) + "..."
