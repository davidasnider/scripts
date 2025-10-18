from __future__ import annotations

import importlib
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


@pytest.fixture
def access_module(monkeypatch):
    tables: list[object] = []

    class FakeTable:
        def __init__(self, name: str, rows: list[dict]):
            self.name = name
            self._rows = rows

        def to_dicts(self):
            return list(self._rows)

    class FakeParser:
        paths: list[str] = []
        catalog_map: dict[str, int] = {}
        catalog_data: dict[str, dict[str, list[object]]] = {}

        def __init__(self, path: str):
            self.path = path
            self.__class__.paths.append(path)

        def tables(self):
            return list(tables)

        @property
        def catalog(self):
            return self.__class__.catalog_map

        def parse_table(self, table_name: str):
            return self.__class__.catalog_data.get(table_name)

    fake_module = ModuleType("access_parser")
    fake_module.AccessParser = FakeParser
    monkeypatch.setitem(sys.modules, "access_parser", fake_module)

    module = importlib.import_module("src.access_analysis")
    module = importlib.reload(module)

    tables.clear()
    FakeParser.paths = []
    FakeParser.catalog_map = {}
    FakeParser.catalog_data = {}

    return module, tables, FakeTable, FakeParser


def test_load_access_tables_returns_dataframes(access_module):
    module, tables, FakeTable, FakeParser = access_module
    tables[:] = [
        FakeTable("Customers", [{"name": "Alice", "city": "Denver"}]),
        FakeTable("Orders", [{"order_id": 1, "amount": 50.0}]),
    ]

    loaded = module.load_access_tables("legacy.mdb")

    assert FakeParser.paths == ["legacy.mdb"]
    assert set(loaded.keys()) == {"Customers", "Orders"}
    assert isinstance(loaded["Customers"], pd.DataFrame)
    assert loaded["Customers"].iloc[0]["name"] == "Alice"


def test_load_access_tables_raises_when_empty(access_module):
    module, tables, FakeTable, _ = access_module
    tables[:] = []

    with pytest.raises(module.AccessAnalysisError):
        module.load_access_tables("empty.mdb")


def test_load_access_tables_supports_catalog_fallback(access_module):
    module, tables, _, FakeParser = access_module
    tables[:] = []
    FakeParser.catalog_map = {"Customers": 10, "MSysObjects": 1}
    FakeParser.catalog_data = {
        "Customers": {"name": ["Alice"], "city": ["Denver"]},
        "MSysObjects": {"Name": ["System"]},
    }

    loaded = module.load_access_tables("catalog-only.mdb")

    assert set(loaded.keys()) == {"Customers"}
    assert loaded["Customers"].iloc[0]["city"] == "Denver"


def test_analyze_text_tables_extracts_themes_and_entities(access_module):
    module, _, _, _ = access_module
    df = pd.DataFrame(
        {
            "notes": [
                "Strong growth from London Markets this quarter.",
                "Concern remains about logistics but overall positive outlook.",
            ]
        }
    )

    results = module.analyze_text_tables({"Insights": df})

    assert results[0].table_name == "Insights"
    assert "growth" in results[0].key_themes
    assert "London Markets" in results[0].named_entities
    assert results[0].sentiment.label == "positive"


def test_analyze_financial_tables_computes_metrics_and_trends(access_module):
    module, _, _, _ = access_module
    df = pd.DataFrame(
        {
            "Revenue": [1000, 1500, 1200],
            "Expenses": [400, 600, 500],
            "Date": [
                "2024-01-15",
                "2024-02-15",
                "2024-02-28",
            ],
            "Notes": ["Initial", "Expansion", "Stabilizing"],
        }
    )

    results = module.analyze_financial_tables({"Ledger": df})

    assert results[0].table_name == "Ledger"
    metric_columns = {metric.column for metric in results[0].metrics}
    assert "Revenue" in metric_columns
    assert results[0].metrics[0].total > 0
    assert results[0].trends
    first_trend = results[0].trends[0]
    assert first_trend.frequency == "monthly"
    assert all(point.total >= 0 for point in first_trend.points)


def test_analyze_access_database_accepts_file_like(access_module, monkeypatch):
    module, tables, FakeTable, FakeParser = access_module
    tables[:] = [
        FakeTable(
            "Activity",
            [
                {"description": "Improved profit margins", "amount": 200},
                {"description": "Risk of decline", "amount": 50},
            ],
        )
    ]

    deleted_paths: list[Path] = []

    def fake_unlink(path):
        deleted_paths.append(Path(path))

    monkeypatch.setattr(module.os, "unlink", fake_unlink)

    result = module.analyze_access_database(
        BytesIO(b"fake-bytes"), filename="legacy.mdb"
    )

    assert isinstance(result, module.AccessAnalysisResult)
    assert FakeParser.paths  # parser invoked with temporary file path
    assert deleted_paths  # temporary file cleaned up
    assert result.table_text["Activity"]
    assert "Improved profit margins" in result.table_text["Activity"]
    assert "Improved profit margins" in result.combined_text
