"""Extracción de mapeos Q→QN desde datos reales."""

from __future__ import annotations

from typing import Any

import pandas as pd

from frs.constants import QN_COLUMNS, QN_PARENT_MAP


def extract_qn_mappings(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Extrae tablas determinísticas Q→QN.

    Returns:
        dict[QN_col, dict[q_value_str, qn_value]]
    """
    mappings: dict[str, dict[str, Any]] = {}

    for qn_col in QN_COLUMNS:
        q_col = QN_PARENT_MAP[qn_col]
        if q_col not in df.columns or qn_col not in df.columns:
            continue

        table: dict[str, Any] = {}
        subset = df[[q_col, qn_col]].dropna()
        for q_val, qn_val in zip(subset[q_col], subset[qn_col]):
            key = _key(q_val)
            val = _to_python(qn_val)
            if key in table and table[key] != val:
                raise ValueError(
                    f"Conflicto en mapeo {q_col}→{qn_col}: "
                    f"{key} mapea a {table[key]} y {val}"
                )
            table[key] = val
        mappings[qn_col] = table

    return mappings


def apply_qn_recodes(df: pd.DataFrame, mappings: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Aplica recodificaciones Q→QN al DataFrame."""
    out = df.copy()

    for qn_col, table in mappings.items():
        q_col = QN_PARENT_MAP[qn_col]
        if q_col not in out.columns:
            continue
        if qn_col not in out.columns:
            out[qn_col] = pd.NA

        def _recode(q_val: Any) -> Any:
            if pd.isna(q_val):
                return pd.NA
            key = _key(q_val)
            if key in table:
                return table[key]
            return pd.NA

        out[qn_col] = out[q_col].apply(_recode)

    return out


def _key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _to_python(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float) and value == int(value):
        return int(value)
    return value
