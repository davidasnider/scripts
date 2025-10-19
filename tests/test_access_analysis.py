import os

import pandas as pd

from src import access_analysis as aa


def _sample_tables():
    return {
        "Accounts": pd.DataFrame(
            {
                "Name": ["Alpha Corp", "Beta LLC"],
                "Amount": [100.0, 250.5],
                "CreatedDate": ["2024-01-15", "2024-02-20"],
                "Notes": [
                    "Strong profit growth reported by Alpha leadership.",
                    "Risk of loss flagged by Beta management.",
                ],
            }
        )
    }


def test_flatten_and_extract_text_helpers():
    tables = _sample_tables()
    frame = tables["Accounts"]

    flattened = aa._flatten_text_columns(frame)
    assert "Alpha Corp" in flattened
    assert "250.5" not in flattened  # numeric columns ignored

    extracted = aa._extract_table_text(frame)
    assert "Name: Alpha Corp Beta LLC" in extracted
    assert "Amount: 100.0 250.5" in extracted

    table_text, combined = aa._gather_table_text(tables)
    assert set(table_text.keys()) == {"Accounts"}
    assert "Table Accounts" in combined


def test_summary_and_theme_helpers():
    text = (
        "This quarter delivered strong growth and profit. "
        "Revenue increase exceeded expectations. "
        "The team is optimistic about future success."
    )
    summary = aa._derive_summary(text, limit=80)
    assert summary.endswith(".")

    themes = aa._extract_key_themes(text)
    assert "growth" in themes
    assert "profit" in themes

    entities = aa._extract_named_entities(
        "Reported by Alice Johnson and Bob Smith in New York."
    )
    assert "Alice Johnson" in entities
    assert "Bob Smith" in entities
    assert "New York" in entities

    sentiment_positive = aa._analyze_sentiment(text)
    assert sentiment_positive.label == "positive"

    sentiment_negative = aa._analyze_sentiment("Loss and decline hurt profit.")
    assert sentiment_negative.label == "negative"


def test_analyze_text_and_financial_tables_generates_results():
    tables = _sample_tables()
    text_results = aa.analyze_text_tables(tables)
    assert len(text_results) == 1
    assert text_results[0].table_name == "Accounts"
    assert text_results[0].sentiment.label in {"positive", "neutral", "negative"}

    financial_results = aa.analyze_financial_tables(tables)
    assert len(financial_results) == 1
    metrics = financial_results[0].metrics
    assert any(metric.column == "Amount" for metric in metrics)
    trend_entries = financial_results[0].trends
    assert trend_entries  # monthly aggregation detected from CreatedDate


def test_find_date_series_detects_string_dates():
    frame = pd.DataFrame(
        {
            "custom_modified": ["2023-01-01", "invalid", "2023-03-15"],
            "value": [5, 6, 7],
        }
    )
    column, series = aa._find_date_series(frame)
    assert column == "custom_modified"
    assert series.notna().sum() == 2


def test_ensure_path_from_upload_bytes(tmp_path):
    payload = b"fake-access-db"
    path = aa._ensure_path_from_upload(payload, filename="db.mdb")
    try:
        assert path.exists()
        assert path.suffix == ".mdb"
        assert path.read_bytes() == payload
    finally:
        if path.exists():
            path.unlink()


def test_analyze_access_database_pipeline(monkeypatch):
    tables = _sample_tables()

    def _fake_load(path):
        assert os.path.exists(path)
        return tables

    monkeypatch.setattr(aa, "load_access_tables", _fake_load)

    result = aa.analyze_access_database(b"binary-db", filename="demo.mdb")

    assert set(result.tables.keys()) == {"Accounts"}
    assert result.table_text["Accounts"]
    assert result.combined_text.startswith("Table Accounts")
    # Confirm helper outputs roundtrip through pipeline
    assert result.text_analysis[0].table_name == "Accounts"
    assert result.financial_analysis[0].table_name == "Accounts"
