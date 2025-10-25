from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Status constants from main.py
PENDING_EXTRACTION = "pending_extraction"
PENDING_ANALYSIS = "pending_analysis"
COMPLETE = "complete"
FAILED = "failed"


class AnalysisStatus(str, Enum):
    """Enum for analysis task status."""

    PENDING = "pending"
    COMPLETE = "complete"
    ERROR = "error"


class AnalysisName(str, Enum):
    """Enum for analysis task names."""

    TEXT_ANALYSIS = "text_analysis"
    IMAGE_DESCRIPTION = "image_description"
    PEOPLE_ANALYSIS = "people_analysis"
    VIDEO_SUMMARY = "video_summary"
    FINANCIAL_ANALYSIS = "financial_analysis"
    NSFW_CLASSIFICATION = "nsfw_classification"
    ACCESS_DB_ANALYSIS = "access_db_analysis"
    PASSWORD_DETECTION = "password_detection"
    ESTATE_ANALYSIS = "estate_analysis"


DEFAULT_ANALYSIS_TASK_VERSION = 1

TEXT_ANALYSIS_VERSION = DEFAULT_ANALYSIS_TASK_VERSION
IMAGE_DESCRIPTION_VERSION = DEFAULT_ANALYSIS_TASK_VERSION
PEOPLE_ANALYSIS_VERSION = DEFAULT_ANALYSIS_TASK_VERSION
VIDEO_SUMMARY_VERSION = DEFAULT_ANALYSIS_TASK_VERSION
FINANCIAL_ANALYSIS_VERSION = DEFAULT_ANALYSIS_TASK_VERSION
NSFW_CLASSIFICATION_VERSION = DEFAULT_ANALYSIS_TASK_VERSION
ACCESS_DB_ANALYSIS_VERSION = DEFAULT_ANALYSIS_TASK_VERSION
PASSWORD_DETECTION_VERSION = 3
ESTATE_ANALYSIS_VERSION = DEFAULT_ANALYSIS_TASK_VERSION

ANALYSIS_TASK_VERSIONS: dict[AnalysisName, int] = {
    AnalysisName.TEXT_ANALYSIS: TEXT_ANALYSIS_VERSION,
    AnalysisName.IMAGE_DESCRIPTION: IMAGE_DESCRIPTION_VERSION,
    AnalysisName.PEOPLE_ANALYSIS: PEOPLE_ANALYSIS_VERSION,
    AnalysisName.VIDEO_SUMMARY: VIDEO_SUMMARY_VERSION,
    AnalysisName.FINANCIAL_ANALYSIS: FINANCIAL_ANALYSIS_VERSION,
    AnalysisName.NSFW_CLASSIFICATION: NSFW_CLASSIFICATION_VERSION,
    AnalysisName.ACCESS_DB_ANALYSIS: ACCESS_DB_ANALYSIS_VERSION,
    AnalysisName.PASSWORD_DETECTION: PASSWORD_DETECTION_VERSION,
    AnalysisName.ESTATE_ANALYSIS: ESTATE_ANALYSIS_VERSION,
}


class AnalysisTask(BaseModel):
    """A single analysis task for a file."""

    name: AnalysisName
    status: AnalysisStatus = AnalysisStatus.PENDING
    error_message: str | None = None
    version: int = Field(default=DEFAULT_ANALYSIS_TASK_VERSION, ge=0)


class FileRecord(BaseModel):
    """Schema for a file record in the manifest."""

    file_path: str
    file_name: str
    mime_type: str
    file_size: int
    last_modified: float
    sha256: str
    status: str = PENDING_EXTRACTION
    extracted_text: str | None = None
    extracted_frames: list[str] | None = None
    analysis_tasks: list[AnalysisTask] = Field(default_factory=list)
    summary: str | None = None
    description: str | None = None
    mentioned_people: list[str] = Field(default_factory=list)
    is_nsfw: bool | None = None
    has_financial_red_flags: bool | None = None
    potential_red_flags: list[str] = Field(default_factory=list)
    incriminating_items: list[str] = Field(default_factory=list)
    confidence_score: int | None = None
    contains_password: bool | None = None
    passwords: dict[str, str] = Field(default_factory=dict)
    has_estate_relevant_info: bool | None = None
    estate_information: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    @field_validator("summary", mode="before")
    @classmethod
    def _ensure_summary_str(cls, value: Any) -> str | None:
        # Coerce structured summaries from legacy manifests into JSON strings.
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, sort_keys=True)
            except (TypeError, ValueError):
                return str(value)
        return str(value)
