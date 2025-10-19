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
# considered a date series. Only columns whose names hint at temporal data
# are considered (to avoid repeatedly parsing arbitrary object columns).
DATE_DETECTION_THRESHOLD = 0.5
DATE_COLUMN_KEYWORDS = {
    "date",
    "datetime",
    "time",
    "timestamp",
    "modified",
    "created",
    "updated",
}

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import validated via tests
    from access_parser import AccessParser
except ImportError as exc:  # pragma: no cover - handled in runtime environments
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
SENTIMENT_SCORES = {word: 1 for word in POSITIVE_WORDS} | {
    word: -1 for word in NEGATIVE_WORDS
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
    table_text: dict[str, str]
    combined_text: str


def _iter_table_rows(table: object) -> list[dict]:
    """Best-effort extraction of row dictionaries from an access_parser table."""

    if hasattr(table, "to_dicts"):
        result = table.to_dicts()  # type: ignore[call-arg]
        if isinstance(result, dict):
            return [result]
        return list(result)
    if hasattr(table, "to_dict"):
        result = table.to_dict()  # type: ignore[call-arg]
        if isinstance(result, dict):
            return [result]
        return list(result)
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
            "access_parser not installed. Install with 'pip install access-parser'."
        )

    parser = AccessParser(file_path)
    tables: dict[str, pd.DataFrame] = {}
    found_tables = False

    for table in _iter_tables(parser):
        found_tables = True
        table_name = getattr(table, "name", None)
        if table_name is None and isinstance(table, (list, tuple)) and table:
            potential_name = table[0]
            if isinstance(potential_name, str):
                table_name = potential_name
                table = table[1] if len(table) > 1 else table
        if table_name is None:
            logger.debug("Skipping table without a name: %s", table)
            continue

        rows = _iter_table_rows(table)
        tables[table_name] = pd.DataFrame(rows)
        logger.debug(
            "Loaded table '%s' with %d rows and %d columns",
            table_name,
            len(rows),
            len(tables[table_name].columns),
        )
    if not found_tables:
        catalog = getattr(parser, "catalog", None)
        parse_table = getattr(parser, "parse_table", None)
        if catalog and callable(parse_table):
            try:
                raw_names = list(catalog.keys())  # type: ignore[attr-defined]
            except AttributeError:
                try:
                    raw_names = list(catalog)
                except TypeError:
                    raw_names = []
            for raw_name in raw_names:
                table_name = str(raw_name)
                if table_name.lower().startswith("msys"):
                    logger.debug("Skipping Access system table '%s'", table_name)
                    continue
                try:
                    table_data = parse_table(table_name)
                except (
                    TypeError,
                    ValueError,
                    KeyError,
                ) as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to parse table '%s' via catalog fallback: %s",
                        table_name,
                        exc,
                    )
                    continue
                if not table_data:
                    logger.debug(
                        "Catalog fallback returned no data for table '%s'", table_name
                    )
                    continue
                try:
                    frame = pd.DataFrame(dict(table_data))
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Could not convert table '%s' to DataFrame: %s",
                        table_name,
                        exc,
                    )
                    continue
                tables[table_name] = frame
                logger.debug(
                    "Loaded table '%s' (%d rows, %d columns) via catalog fallback",
                    table_name,
                    len(frame),
                    len(frame.columns),
                )

    if not tables:
        raise AccessAnalysisError("No tables were discovered in the Access database.")

    return tables


def _flatten_text_columns(df: pd.DataFrame) -> str:
    text_columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
    if not text_columns:
        return ""

    parts: list[str] = []
    for col in text_columns:
        series = df[col].dropna().astype(str)
        parts.append(" ".join(series.tolist()))
    return " \n".join(parts)


def _extract_table_text(df: pd.DataFrame) -> str:
    """Flatten all column values into a single text block."""

    if df.empty:
        return ""

    sections: list[str] = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        values = series.astype(str).tolist()
        if not values:
            continue
        column_name = str(column)
        sections.append(f"{column_name}: {' '.join(values)}")
    return "\n".join(sections)


def _gather_table_text(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, str], str]:
    """Build per-table and combined text representations."""

    table_text: dict[str, str] = {}
    combined_chunks: list[str] = []

    for name, df in tables.items():
        text = _extract_table_text(df)
        table_text[name] = text
        if text:
            combined_chunks.append(f"Table {name}:\n{text}")

    combined_text = "\n\n".join(combined_chunks)
    return table_text, combined_text


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
    total = len(tokens) if tokens else 1
    score = sum(SENTIMENT_SCORES.get(token, 0) for token in tokens)
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
            column_lower = str(column).lower()
            if any(keyword in column_lower for keyword in DATE_COLUMN_KEYWORDS):
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
        if hasattr(file_reference, "seek"):
            try:
                file_reference.seek(0)
            except (OSError, AttributeError) as exc:
                logger.debug("Failed to rewind uploaded file-like object: %s", exc)
        data = file_reference.read()  # type: ignore[assignment]
    if not data:
        raise AccessAnalysisError("Uploaded file is empty.")

    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as temp_file:
        temp_file.write(data)
    if not isinstance(file_reference, bytes) and hasattr(file_reference, "seek"):
        try:
            file_reference.seek(0)
        except (OSError, AttributeError) as exc:
            logger.debug(
                "Failed to restore position on uploaded file-like object: %s", exc
            )
    return Path(temp_path)


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
        table_text, combined_text = _gather_table_text(tables)
        text_analysis = analyze_text_tables(tables)
        financial_analysis = analyze_financial_tables(tables)
        return AccessAnalysisResult(
            tables=tables,
            text_analysis=text_analysis,
            financial_analysis=financial_analysis,
            table_text=table_text,
            combined_text=combined_text,
        )
    finally:
        if created_temp:
            try:
                os.unlink(path)
            except OSError:
                logger.debug("Failed to remove temporary file: %s", path)
