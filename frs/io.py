"""Carga y exportación de datos de encuesta."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from frs.constants import ALL_COLUMNS, SPSS_MISSING_STR, is_spss_missing

_EXPORT_FORMAT_KEYS = (
    "export_quoted",
    "export_decimal_places",
    "export_as_integer",
    "export_strip_trailing_zeros",
)


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


def _iter_csv_fields(line: str) -> list[tuple[bool, str]]:
    """Divide una línea CSV devolviendo (quoted, valor) por campo."""
    fields: list[tuple[bool, str]] = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == '"':
            quoted = True
            i += 1
            start = i
            while i < n:
                if line[i] == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        i += 2
                        continue
                    break
                i += 1
            value = line[start:i].replace('""', '"')
            fields.append((quoted, value))
            i += 1
            if i < n and line[i] == ",":
                i += 1
        else:
            start = i
            while i < n and line[i] != ",":
                i += 1
            value = line[start:i].rstrip("\r\n")
            fields.append((False, value))
            if i < n and line[i] == ",":
                i += 1
    return fields


def _count_decimal_places(text: str) -> int | None:
    text = text.strip()
    if not text or "e" in text.lower():
        return None
    if "." not in text:
        return 0
    fractional = text.split(".", 1)[1]
    if not fractional.isdigit():
        return None
    return len(fractional)


def _looks_integer(text: str) -> bool:
    text = text.strip()
    if not text or is_spss_missing(text):
        return False
    if "e" in text.lower():
        return False
    if "." in text:
        return False
    try:
        int(text)
        return True
    except ValueError:
        return False


def infer_csv_export_format(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    n_sample: int = 500,
) -> dict[str, dict[str, Any]]:
    """Infiere formato textual de exportación desde filas crudas del CSV."""
    path = Path(path)
    columns = columns or ALL_COLUMNS
    stats: dict[str, dict[str, Any]] = {
        col: {
            "quoted_any": False,
            "decimal_places_max": 0,
            "has_decimal": False,
            "integer_only": True,
            "strip_trailing_zeros": False,
            "seen_fractional_lengths": set(),
        }
        for col in columns
    }

    with path.open(encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            return {}

        header_fields = [name for _, name in _iter_csv_fields(header_line.rstrip("\r\n"))]
        col_index = {name: idx for idx, name in enumerate(header_fields)}

        for _ in range(n_sample):
            line = f.readline()
            if not line:
                break
            fields = _iter_csv_fields(line.rstrip("\r\n"))
            for col in columns:
                idx = col_index.get(col)
                if idx is None or idx >= len(fields):
                    continue
                quoted, raw = fields[idx]
                raw = raw.strip()
                if not raw or is_spss_missing(raw):
                    continue

                col_stats = stats[col]
                if quoted:
                    col_stats["quoted_any"] = True

                if _looks_integer(raw):
                    continue

                col_stats["integer_only"] = False
                dp = _count_decimal_places(raw)
                if dp is None:
                    continue
                if dp > 0:
                    col_stats["has_decimal"] = True
                    col_stats["decimal_places_max"] = max(col_stats["decimal_places_max"], dp)
                    col_stats["seen_fractional_lengths"].add(dp)

    result: dict[str, dict[str, Any]] = {}
    for col, col_stats in stats.items():
        spec: dict[str, Any] = {}
        if col_stats["quoted_any"]:
            spec["export_quoted"] = True
        if col_stats["integer_only"]:
            spec["export_as_integer"] = True
        elif col_stats["has_decimal"]:
            spec["export_decimal_places"] = col_stats["decimal_places_max"]
            lengths = col_stats["seen_fractional_lengths"]
            if len(lengths) > 1 or (
                col_stats["decimal_places_max"] > 0
                and min(lengths) < col_stats["decimal_places_max"]
            ):
                spec["export_strip_trailing_zeros"] = True
        if spec:
            result[col] = spec
    return result


def _export_spec_from_column(col: str, col_schema: dict[str, Any] | None) -> dict[str, Any]:
    if not col_schema:
        return {}
    return {k: col_schema[k] for k in _EXPORT_FORMAT_KEYS if k in col_schema}


def _resolve_export_specs(
    schema: dict[str, Any] | None,
    reference_path: str | Path | None,
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    if schema:
        for col, col_schema in schema.get("columns", {}).items():
            spec = _export_spec_from_column(col, col_schema)
            if spec:
                specs[col] = spec

    if reference_path and Path(reference_path).exists():
        inferred = infer_csv_export_format(reference_path)
        for col, spec in inferred.items():
            specs.setdefault(col, {}).update(spec)

    return specs


def _to_export_value(value: Any, spec: dict[str, Any] | None = None) -> str:
    spec = spec or {}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return SPSS_MISSING_STR
    if pd.isna(value):
        return SPSS_MISSING_STR
    if isinstance(value, float) and not math.isfinite(value):
        return SPSS_MISSING_STR

    numeric = value
    if not isinstance(numeric, (int, float)):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)

    as_integer = spec.get("export_as_integer", False)
    decimal_places = spec.get("export_decimal_places")
    strip_trailing = spec.get("export_strip_trailing_zeros", False)

    if as_integer or (
        decimal_places is None
        and isinstance(numeric, float)
        and abs(numeric - round(numeric)) < 1e-9
        and abs(numeric) < 1e15
    ):
        return str(int(round(numeric)))

    if decimal_places is not None:
        formatted = f"{float(numeric):.{int(decimal_places)}f}"
        if strip_trailing:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    if isinstance(numeric, float):
        if numeric == int(numeric) and abs(numeric) < 1e15:
            return str(int(numeric))
        text = repr(numeric)
        if "e" in text.lower():
            return text
        return format(numeric, ".15g").rstrip("0").rstrip(".") if "." in format(numeric, ".15g") else format(numeric, ".15g")

    return str(numeric)


def format_for_export(
    df: pd.DataFrame,
    export_specs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Formatea valores para exportación CSV según especificaciones por columna."""
    export_specs = export_specs or {}
    out = df.reindex(columns=ALL_COLUMNS).copy()
    for col in out.columns:
        spec = export_specs.get(col, {})
        out[col] = out[col].apply(lambda v, s=spec: _to_export_value(v, s))
    return out


def restore_spss_missing(
    df: pd.DataFrame,
    export_specs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Reemplaza NaN/NA por sentinel SPSS y aplica formato de exportación."""
    return format_for_export(df, export_specs)


def save_survey(
    df: pd.DataFrame,
    path: str | Path,
    *,
    schema: dict[str, Any] | None = None,
    reference_path: str | Path | None = None,
) -> None:
    """Exporta DataFrame con sentinel SPSS y columnas en orden original."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_specs = _resolve_export_specs(schema, reference_path)
    export_df = restore_spss_missing(df, export_specs)
    export_df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


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
