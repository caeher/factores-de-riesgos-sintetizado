"""Carga y exportación de datos de encuesta."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from frs.constants import ALL_COLUMNS, SPSS_MISSING, SPSS_MISSING_STR, is_spss_missing


def load_survey(path: str | Path) -> pd.DataFrame:
    """Carga CSV reemplazando sentinel SPSS por NaN en memoria."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.reindex(columns=ALL_COLUMNS)

    for col in df.columns:
        mask = df[col].map(is_spss_missing)
        df.loc[mask, col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def restore_spss_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza NaN/NA por el sentinel SPSS para exportación."""
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(_to_export_value)
    return out


def _to_export_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return SPSS_MISSING_STR
    if pd.isna(value):
        return SPSS_MISSING_STR
    if isinstance(value, float):
        if not math.isfinite(value):
            return SPSS_MISSING_STR
        if value == int(value) and abs(value) < 1e15:
            return int(value) if abs(value - int(value)) < 1e-9 else value
        return value
    return value


def save_survey(df: pd.DataFrame, path: str | Path) -> None:
    """Exporta DataFrame con sentinel SPSS y columnas en orden original."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_df = restore_spss_missing(df.reindex(columns=ALL_COLUMNS))
    export_df.to_csv(path, index=False, quoting=1)  # csv.QUOTE_ALL


def load_schema(path: str | Path) -> dict[str, Any]:
    """Carga schema YAML."""
    path = Path(path)
    if not path.exists():
        return {"columns": {}}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "columns" not in data:
        data["columns"] = {}
    return data


def save_schema(schema: dict[str, Any], path: str | Path) -> None:
    """Guarda schema YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(schema, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
