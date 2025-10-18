"""Utilities for ingesting and analyzing Microsoft Access databases."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

# At least 50% of values must be parseable as dates for a column to be
# considered a date series.
DATE_DETECTION_THRESHOLD = 0.5

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import validated via tests
    from access_parser import AccessParser
except Exception as exc:  # pragma: no cover - handled in runtime environments
    AccessParser = None  # type: ignore[assignment]
    logger.debug("access_parser import failed: %s", exc)

POSITIVE_WORDS = {
    "growth",
    "increase",
    "profit",
    "success",
    "improved",
    "strong",
    "gain",
    "positive",
    "benefit",
    "optimistic",
}
NEGATIVE_WORDS = {
    "loss",
    "decline",
    "drop",
    "negative",
    "risk",
    "weak",
    "concern",
    "issue",
    "problem",
    "decrease",
}
STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "have",
    "will",
    "shall",
    "into",
    "about",
    "there",
    "their",
    "over",
    "under",
    "while",
    "because",
    "were",
    "been",
    "being",
    "after",
    "before",
    "which",
    "upon",
    "during",
    "without",
    "within",
    "between",
    "through",
}
FINANCIAL_KEYWORDS = {
    "amount",
    "balance",
    "cost",
    "expense",
    "income",
    "price",
    "profit",
    "revenue",
    "sales",
    "total",
    "qty",
    "quantity",
}


class AccessAnalysisError(RuntimeError):
    """Raised when a Microsoft Access database cannot be analyzed."""


@dataclass(slots=True)
class SentimentResult:
    """Simple sentiment outcome for a text corpus."""

    label: str
    score: float


@dataclass(slots=True)
class TextAnalysisResult:
    """Summary insights for a table's text columns."""

    table_name: str
    summary: str
    key_themes: list[str]
    sentiment: SentimentResult
    named_entities: list[str]


@dataclass(slots=True)
class FinancialMetric:
    """Aggregated metric for a numeric column."""

    column: str
    total: float
    average: float
    minimum: float
    maximum: float


@dataclass(slots=True)
class FinancialTrendPoint:
    """Time-series aggregate for a numeric column."""

    period: str
    total: float


@dataclass(slots=True)
class FinancialTrend:
    """Collection of trend points for a numeric column."""

    column: str
    frequency: str
    points: list[FinancialTrendPoint]


@dataclass(slots=True)
class FinancialAnalysisResult:
    """Financial insights for a single table."""

    table_name: str
    metrics: list[FinancialMetric]
    trends: list[FinancialTrend]
    record_count: int


@dataclass(slots=True)
class AccessAnalysisResult:
    """Container for the full Access database analysis."""

    tables: dict[str, pd.DataFrame]
    text_analysis: list[TextAnalysisResult]
    financial_analysis: list[FinancialAnalysisResult]


def _iter_table_rows(table: object) -> Iterable[dict]:
    """Best-effort extraction of row dictionaries from an access_parser table."""

    if hasattr(table, "to_dicts"):
        return list(table.to_dicts())  # type: ignore[call-arg]
    if hasattr(table, "to_dict"):
        return list(table.to_dict())  # type: ignore[call-arg]
    if hasattr(table, "rows"):
        rows = getattr(table, "rows")
        if callable(rows):
            return list(rows())
        return list(rows)
    if hasattr(table, "records"):
        records = getattr(table, "records")
        if callable(records):
            return list(records())
        return list(records)
    return list(table)  # type: ignore[arg-type]


def _iter_tables(parser: object) -> Iterable[object]:
    if hasattr(parser, "tables"):
        tables = getattr(parser, "tables")
        return tables() if callable(tables) else tables
    if hasattr(parser, "iter_tables"):
        return parser.iter_tables()
    if hasattr(parser, "get_tables"):
        tables = parser.get_tables()
        return tables() if callable(tables) else tables
    return []


def load_access_tables(file_path: str) -> dict[str, pd.DataFrame]:
    """Load all tables from an Access database into pandas DataFrames."""

    if AccessParser is None:  # pragma: no cover - runtime guard
        raise AccessAnalysisError(
            "access_parser is not installed; cannot load Access databases."
        )

    parser = AccessParser(file_path)
    tables: dict[str, pd.DataFrame] = {}

    for table in _iter_tables(parser):
        table_name = getattr(table, "name", None)
        if table_name is None and isinstance(table, (list, tuple)) and table:
            potential_name = table[0]
            if isinstance(potential_name, str):
                table_name = potential_name
                table = table[1] if len(table) > 1 else table
        if table_name is None:
            logger.debug("Skipping table without a name: %s", table)
            continue

        rows = list(_iter_table_rows(table))
        tables[table_name] = pd.DataFrame(rows)
        logger.debug(
            "Loaded table '%s' with %d rows and %d columns",
            table_name,
            len(rows),
            len(tables[table_name].columns),
        )

    if not tables:
        raise AccessAnalysisError("No tables were discovered in the Access database.")

    return tables


def _flatten_text_columns(df: pd.DataFrame) -> str:
    text_columns = [
        col
        for col in df.columns
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])
    ]
    if not text_columns:
        return ""

    parts: list[str] = []
    for col in text_columns:
        series = df[col].dropna().astype(str)
        parts.append(" ".join(series.tolist()))
    return " \n".join(parts)


def _derive_summary(text: str, limit: int = 400) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "No textual content detected."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    summary = " ".join(sentences[:2])

    if len(summary) > limit:
        summary = summary[: limit - 3].rstrip() + "..."
    return summary


def _extract_key_themes(text: str, limit: int = 5) -> list[str]:
    words = [re.sub(r"[^a-zA-Z]", "", token.lower()) for token in text.split() if token]
    filtered = [
        word for word in words if word and word not in STOP_WORDS and len(word) > 3
    ]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(limit)]


def _extract_named_entities(text: str, limit: int = 10) -> list[str]:
    pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
    entities = []
    seen = set()
    for match in pattern.findall(text):
        normalized = match.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            entities.append(normalized)
        if len(entities) >= limit:
            break
    return entities


def _analyze_sentiment(text: str) -> SentimentResult:
    if not text.strip():
        return SentimentResult(label="neutral", score=0.0)

    tokens = [
        re.sub(r"[^a-zA-Z]", "", token.lower()) for token in text.split() if token
    ]
    total = len(tokens) or 1
    score = 0
    for token in tokens:
        if token in POSITIVE_WORDS:
            score += 1
        elif token in NEGATIVE_WORDS:
            score -= 1
    normalized = score / total
    if normalized > 0.05:
        label = "positive"
    elif normalized < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return SentimentResult(label=label, score=normalized)


def analyze_text_tables(tables: dict[str, pd.DataFrame]) -> list[TextAnalysisResult]:
    """Run lightweight NLP analysis across all text columns."""

    results: list[TextAnalysisResult] = []
    for name, df in tables.items():
        text_blob = _flatten_text_columns(df)
        summary = _derive_summary(text_blob)
        key_themes = _extract_key_themes(text_blob)
        named_entities = _extract_named_entities(text_blob)
        sentiment = _analyze_sentiment(text_blob)
        results.append(
            TextAnalysisResult(
                table_name=name,
                summary=summary,
                key_themes=key_themes,
                sentiment=sentiment,
                named_entities=named_entities,
            )
        )
    return results


def _find_date_series(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    for column in df.columns:
        series = df[column]
        if is_datetime64_any_dtype(series):
            return column, series
        if series.dtype == "object":
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().sum() >= max(
                1, int(len(series) * DATE_DETECTION_THRESHOLD)
            ):
                return column, parsed
    return None, None


def analyze_financial_tables(
    tables: dict[str, pd.DataFrame],
) -> list[FinancialAnalysisResult]:
    """Inspect numeric columns for financial performance signals."""

    insights: list[FinancialAnalysisResult] = []

    for name, df in tables.items():
        numeric_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if is_numeric_dtype(df[col]) or any(
                keyword in col_lower for keyword in FINANCIAL_KEYWORDS
            ):
                numeric_cols.append(col)
        metrics: list[FinancialMetric] = []
        trends: list[FinancialTrend] = []

        if numeric_cols:
            for col in numeric_cols:
                series = pd.to_numeric(df[col], errors="coerce")
                valid = series.dropna()
                if valid.empty:
                    continue
                metrics.append(
                    FinancialMetric(
                        column=str(col),
                        total=float(valid.sum()),
                        average=float(valid.mean()),
                        minimum=float(valid.min()),
                        maximum=float(valid.max()),
                    )
                )

            _, date_series = _find_date_series(df)
            if date_series is not None:
                date_series = pd.to_datetime(date_series, errors="coerce")
                if date_series.notna().any():
                    frame = df.copy()
                    frame["__analysis_date__"] = date_series
                    frame = frame.dropna(subset=["__analysis_date__"])
                    frame["__period__"] = frame["__analysis_date__"].dt.to_period("M")

                    for metric in metrics:
                        numeric_values = pd.to_numeric(
                            frame[metric.column], errors="coerce"
                        )
                        grouped = (
                            frame.assign(__value__=numeric_values)
                            .dropna(subset=["__value__"])
                            .groupby("__period__")["__value__"]
                            .sum()
                        )
                        points = [
                            FinancialTrendPoint(period=str(period), total=float(total))
                            for period, total in grouped.items()
                        ]
                        if points:
                            trends.append(
                                FinancialTrend(
                                    column=metric.column,
                                    frequency="monthly",
                                    points=points,
                                )
                            )

        insights.append(
            FinancialAnalysisResult(
                table_name=name,
                metrics=metrics,
                trends=trends,
                record_count=len(df),
            )
        )

    return insights


def _ensure_path_from_upload(
    file_reference: str | os.PathLike[str] | BinaryIO | bytes,
    *,
    filename: str | None = None,
) -> Path:
    if isinstance(file_reference, (str, os.PathLike)):
        return Path(file_reference)

    suffix = Path(filename or "upload.mdb").suffix or ".mdb"

    if isinstance(file_reference, bytes):
        data = file_reference
    else:
        data = file_reference.read()  # type: ignore[assignment]
    if not data:
        raise AccessAnalysisError("Uploaded file is empty.")

    temp_file = tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False)
    temp_file.write(data)
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def analyze_access_database(
    file_reference: str | os.PathLike[str] | BinaryIO | bytes,
    *,
    filename: str | None = None,
) -> AccessAnalysisResult:
    """End-to-end pipeline for Access ingestion and analysis."""

    path = _ensure_path_from_upload(file_reference, filename=filename)
    created_temp = not isinstance(file_reference, (str, os.PathLike))

    try:
        tables = load_access_tables(str(path))
        text_analysis = analyze_text_tables(tables)
        financial_analysis = analyze_financial_tables(tables)
        return AccessAnalysisResult(
            tables=tables,
            text_analysis=text_analysis,
            financial_analysis=financial_analysis,
        )
    finally:
        if created_temp:
            try:
                os.unlink(path)
            except OSError:
                logger.debug("Failed to remove temporary file: %s", path)
