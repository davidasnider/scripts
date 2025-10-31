"""Standardized prompts for AI analysis tasks."""

from __future__ import annotations

JSON_RESPONSE_FORMAT = "json"
JSON_OUTPUT_OPTIONS = {"temperature": 0}
STRICT_JSON_REMINDER = (
    "REMINDER: Reply with strictly valid JSON. Use double quotes for all keys and "
    "string values, include commas between fields, and do not add commentary."
)

# --- Text Analysis Prompts ---

TEXT_ANALYSIS_SYSTEM_PROMPT = """You are a document analyst. Analyze the provided text
and return a JSON response with two keys: "summary" (a concise summary paragraph)
and "mentioned_people" (a list of names, excluding usernames).

Example response:
{
  "summary": "Concise paragraph here.",
  "mentioned_people": ["Alice Johnson", "Robert Smith"]
}

Respond only with valid JSON. Do not wrap the JSON in code blocks or backticks."""


def build_text_analysis_prompt(text: str) -> tuple[str, str]:
    """Builds the system and user prompts for single-chunk text analysis."""
    return TEXT_ANALYSIS_SYSTEM_PROMPT, text


def build_text_analysis_chunk_prompt(
    *, chunk: str, index: int, chunk_count: int
) -> tuple[str, str]:
    """Builds the system and user prompts for chunked text analysis."""
    system_prompt = (
        f"You are a document analyst. Analyze chunk {index}/{chunk_count} of the "
        'provided text and return a JSON response with two keys: "summary" (a '
        'concise summary of the chunk) and "mentioned_people" (a list of names, '
        "excluding usernames).\n\n"
        "Example response:\n"
        "{\n"
        '  "summary": "Concise paragraph for this chunk.",\n'
        '  "mentioned_people": ["Alice Johnson"]\n'
        "}\n\n"
        "Respond only with valid JSON. "
        "Do not wrap the JSON in code blocks or backticks."
    )
    return system_prompt, chunk


# --- Password Detection Prompts ---

PASSWORD_DETECTOR_SYSTEM_PROMPT = """You are a security auditor. Find passwords, 
secrets, tokens, or keys in the text. Base your decision on context ('password', 
'secret', etc.).

Respond with a JSON object with a 'passwords' field, a list of objects each with 
'context' and 'password' keys.

CRITICAL REMINDER: The JSON example is for structure only. Do NOT include the 
example data in your response. Your response must only contain information from 
the provided text.

Example response:
{
  "passwords": [
    {
      "context": "Login credential for example.com",
      "password": "user_password123"
    }
  ]
}

If no credentials are found, return `{"passwords": []}`. Respond with raw JSON only."""


def build_password_prompt(text: str) -> tuple[str, str]:
    """Builds the system and user prompts for single-chunk password detection."""
    return PASSWORD_DETECTOR_SYSTEM_PROMPT, text


def build_password_chunk_prompt(
    *, chunk: str, index: int, chunk_count: int
) -> tuple[str, str]:
    """Builds the system and user prompts for chunked password detection."""
    system_prompt = (
        "You are a security auditor. Find passwords, secrets, tokens, or keys in "
        f"chunk {index}/{chunk_count} of the provided text. Base your decision on "
        "context ('password', 'secret', etc.).\n\n"
        "Respond with a JSON object with a 'passwords' field, a list of objects "
        "each with 'context' and 'password' keys.\n\n"
        "CRITICAL REMINDER: The JSON example is for structure only. Do NOT include "
        "the example data in your response. Your response must only contain "
        "information from the provided text.\n\n"
        "Example response:\n"
        "{\n"
        '  "passwords": [\n'
        "    {\n"
        '      "context": "Login credential for example.com",\n'
        '      "password": "user_password123"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        'If no credentials are found, return `{"passwords": []}`. Respond with raw '
        "JSON only."
    )
    return system_prompt, chunk


# --- Estate Analysis Prompts ---

ESTATE_CATEGORY_KEYS = [
    "Legal",
    "Financial",
    "Insurance",
    "Digital",
    "Medical",
    "Personal",
]
ESTATE_ANALYSIS_SYSTEM_PROMPT = f"""You are an estate planning assistant. 
Extract actionable information for an executor from the text. 
Focus on legal documents, financial accounts, property, debts, and digital assets.

CRITICAL REMINDER: The JSON examples are for structure only. 
Do NOT include data from the examples in your response. 
Your response must only contain information from the provided text.

Return a JSON object with these top-level keys: {', '.join(ESTATE_CATEGORY_KEYS)}. 
Each key must map to a list of objects with "item", "why_it_matters", and 
"details" keys. Omit entries with placeholder values. If no information is found, 
return an empty JSON object {{}}.

Example response:
{{
  "Legal": [
    {{
      "item": "Last will and testament",
      "why_it_matters": "Names executor and divides property",
      "details": "Stored in the safe at 123 Main St., combination 1965"
    }}
  ],
  "Financial": [
    {{
      "item": "Chase savings account",
      "why_it_matters": "Funds available for estate expenses",
      "details": "Account ending 1234, branch on 5th Ave"
    }}
  ]
}}

Respond only with raw JSON."""

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
ESTATE_MIN_FIELDS_PER_ENTRY = 2
ESTATE_SKIP_BARE_STRING_LOG = (
    "Skipping estate entry emitted as bare string; expected structured object."
)


def build_estate_analysis_prompt(text: str) -> tuple[str, str]:
    """Builds the system and user prompts for single-chunk estate analysis."""
    return ESTATE_ANALYSIS_SYSTEM_PROMPT, text


def build_estate_analysis_chunk_prompt(
    *, chunk: str, index: int, chunk_count: int, source_name: str
) -> tuple[str, str]:
    """Builds the system and user prompts for chunked estate analysis."""
    system_prompt = (
        f"You are an estate planning assistant. Extract actionable information for an "
        f"executor from chunk {index}/{chunk_count} of {source_name}. Focus on legal "
        "documents, financial accounts, property, debts, and digital assets.\n\n"
        "CRITICAL REMINDER: The JSON examples are for structure only. Do NOT "
        "include data from the examples in your response. Your response must "
        "only contain information from the provided text.\n\n"
        "Return a JSON object with these top-level keys: "
        f"{', '.join(ESTATE_CATEGORY_KEYS)}. Each key must map to a list of "
        'objects with "item", "why_it_matters", and "details" keys. Omit entries '
        "with placeholder values. If no information is found, return an empty JSON "
        "object {}.\n\n"
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
        "Respond only with raw JSON."
    )
    return system_prompt, chunk


# --- Financial Analysis Prompts ---

FINANCIAL_ANALYSIS_SYSTEM_PROMPT = """You are a forensic accountant. 
Analyze the financial text and return a JSON response with "summary", 
"potential_red_flags", "incriminating_items", and "confidence_score" (0-100).

Example response:
{
  "summary": "Overall summary of the document.",
  "potential_red_flags": ["Late payment noted"],
  "incriminating_items": ["Unreported cash deposit"],
  "confidence_score": 78
}

Respond only with valid JSON."""


def build_financial_analysis_prompt(text: str) -> tuple[str, str]:
    """Builds the system and user prompts for single-chunk financial analysis."""
    return FINANCIAL_ANALYSIS_SYSTEM_PROMPT, text


def build_financial_analysis_chunk_prompt(
    *, chunk: str, index: int, chunk_count: int
) -> tuple[str, str]:
    """Builds the system and user prompts for chunked financial analysis."""
    system_prompt = (
        f"You are a forensic accountant. Analyze chunk {index}/{chunk_count} of the "
        'financial text and return a JSON response with "summary", '
        '"potential_red_flags", "incriminating_items", and "confidence_score" '
        "(0-100).\n\n"
        "Example response:\n"
        "{\n"
        '  "summary": "Chunk-specific summary.",\n'
        '  "potential_red_flags": [],\n'
        '  "incriminating_items": [],\n'
        '  "confidence_score": 60\n'
        "}\n\n"
        "Respond only with valid JSON."
    )
    return system_prompt, chunk
