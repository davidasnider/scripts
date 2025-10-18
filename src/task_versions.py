from __future__ import annotations

from src.schema import AnalysisName

# Version definitions for each analysis task
# Increment a task's version number when its logic changes to trigger re-analysis.
TASK_VERSIONS = {
    AnalysisName.TEXT_SUMMARY_ANALYSIS: 1,
    AnalysisName.PEOPLE_DETECTION_ANALYSIS: 1,
    AnalysisName.IMAGE_DESCRIPTION: 1,
    AnalysisName.VIDEO_SUMMARY: 1,
    AnalysisName.FINANCIAL_ANALYSIS: 1,
    AnalysisName.NSFW_CLASSIFICATION: 1,
}