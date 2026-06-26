"""Tests de carga y exportación CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from frs.io import (
    _iter_csv_fields,
    _to_export_value,
    infer_csv_export_format,
    save_survey,
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "sample.csv"
    path.write_text(
        "Q1,Q4,weight\n"
        "1,1.65,114.8873\n"
        "2,1.6,200.5\n",
        encoding="utf-8",
    )
    return path


def test_iter_csv_fields_unquoted():
    fields = _iter_csv_fields("1,1.65,114.8873")
    assert fields == [(False, "1"), (False, "1.65"), (False, "114.8873")]


def test_iter_csv_fields_quoted():
    fields = _iter_csv_fields('"1","1.65","114.8873"')
    assert fields == [(True, "1"), (True, "1.65"), (True, "114.8873")]


def test_infer_csv_export_format(sample_csv: Path):
    fmt = infer_csv_export_format(sample_csv, columns=["Q1", "Q4", "weight"])
    assert fmt["Q1"]["export_as_integer"] is True
    assert fmt["Q4"]["export_decimal_places"] == 2
    assert fmt["weight"]["export_decimal_places"] == 4


def test_to_export_value_formats():
    assert _to_export_value(1.0, {"export_as_integer": True}) == "1"
    assert _to_export_value(1.65, {"export_decimal_places": 2}) == "1.65"
    assert _to_export_value(1.60, {"export_decimal_places": 2, "export_strip_trailing_zeros": True}) == "1.6"
    assert _to_export_value(114.8873, {"export_decimal_places": 4}) == "114.8873"


def test_save_survey_no_quote_all(tmp_path: Path):
    df = pd.DataFrame({"Q1": [1, 2], "Q4": [1.65, 1.6], "weight": [114.8873, 200.5]})
    out = tmp_path / "out.csv"
    schema = {
        "columns": {
            "Q1": {"export_as_integer": True},
            "Q4": {"export_decimal_places": 2, "export_strip_trailing_zeros": True},
            "weight": {"export_decimal_places": 4},
        }
    }
    save_survey(df, out, schema=schema)

    text = out.read_text(encoding="utf-8")
    data_line = text.splitlines()[1]
    assert '"1"' not in data_line
    assert ",1.65," in data_line
    assert ",114.8873," in data_line

    with out.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    values = dict(zip(header, rows[1]))
    assert values["Q1"] == "1"
    assert values["Q4"] == "1.65"
    assert values["weight"] == "114.8873"
