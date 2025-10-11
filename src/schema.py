from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# Status constants from main.py
PENDING_EXTRACTION = "pending_extraction"
PENDING_ANALYSIS = "pending_analysis"
COMPLETE = "complete"


class AnalysisStatus(str, Enum):
    """Enum for analysis task status."""

    PENDING = "pending"
    COMPLETE = "complete"
    ERROR = "error"


class AnalysisName(str, Enum):
    """Enum for analysis task names."""

    TEXT_ANALYSIS = "text_analysis"
    IMAGE_DESCRIPTION = "image_description"
    VIDEO_SUMMARY = "video_summary"
    FINANCIAL_ANALYSIS = "financial_analysis"
    NSFW_CLASSIFICATION = "nsfw_classification"


class AnalysisTask(BaseModel):
    """A single analysis task for a file."""

    name: AnalysisName
    status: AnalysisStatus = AnalysisStatus.PENDING
    error_message: str | None = None


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
