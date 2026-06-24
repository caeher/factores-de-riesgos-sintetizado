"""Metadatos SDV para el modelo de síntesis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sdv.metadata import SingleTableMetadata

from frs.constants import CONTINUOUS_COLUMNS, DESIGN_COLUMNS, SYNTH_COLUMNS


def build_sdv_metadata(df: pd.DataFrame, schema: dict[str, Any]) -> SingleTableMetadata:
    """Construye SingleTableMetadata para columnas de capa A."""
    synth_df = df[SYNTH_COLUMNS].copy()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(synth_df)

    col_specs = schema.get("columns", {})
    for col in SYNTH_COLUMNS:
        spec = col_specs.get(col, {})
        dtype = spec.get("dtype")

        if col in CONTINUOUS_COLUMNS or dtype == "numerical":
            metadata.update_column(col, sdtype="numerical")
        elif col in ("stratum", "psu") or dtype == "categorical":
            metadata.update_column(col, sdtype="categorical")
        else:
            metadata.update_column(col, sdtype="categorical")

    return metadata


def prepare_fit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara DataFrame para fit SDV (solo columnas sintetizables)."""
    fit_df = df[SYNTH_COLUMNS].copy()

    for col in fit_df.columns:
        if col in CONTINUOUS_COLUMNS or col == "weight":
            fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce")
        elif col in DESIGN_COLUMNS:
            if col == "weight":
                fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce")
            else:
                fit_df[col] = fit_df[col].apply(_cat_str)
        else:
            fit_df[col] = fit_df[col].apply(_cat_str_or_na)

    return fit_df


def prepare_sample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte salida SDV a tipos utilizables."""
    out = df.copy()
    for col in out.columns:
        if col in CONTINUOUS_COLUMNS or col == "weight":
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif col in DESIGN_COLUMNS and col != "weight":
            out[col] = out[col].apply(_parse_cat)
        else:
            out[col] = out[col].apply(_parse_cat)
    return out


def _cat_str(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _cat_str_or_na(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _parse_cat(value: Any) -> Any:
    if value is None or (isinstance(value, str) and value == "nan"):
        return pd.NA
    if isinstance(value, float) and pd.isna(value):
        return pd.NA
    try:
        f = float(value)
        if f == int(f):
            return int(f)
        return f
    except (TypeError, ValueError):
        return value
