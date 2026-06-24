"""Derivación de indicadores globales."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from frs.constants import DERIVED_COLUMNS, DERIVED_PARENTS


@dataclass
class DerivedRules:
    """Reglas aprendidas para variables derivadas."""

    deterministic: dict[str, dict[tuple[str, ...], Any]] = field(default_factory=dict)
    conditional: dict[str, dict[tuple[str, ...], dict[Any, float]]] = field(default_factory=dict)
    bmi_fallback: dict[str, dict[tuple[str, ...], Any]] = field(default_factory=dict)
    missing_prob: dict[str, dict[tuple[str, ...], float]] = field(default_factory=dict)
    global_missing_prob: dict[str, float] = field(default_factory=dict)
    value_marginal: dict[str, dict[Any, float]] = field(default_factory=dict)
    partial_parent_fill_rate: dict[str, float] = field(default_factory=dict)


def learn_derived_rules(df: pd.DataFrame) -> DerivedRules:
    """Aprende reglas determinísticas y condicionales desde datos reales."""
    rules = DerivedRules()

    for col in DERIVED_COLUMNS:
        if col not in df.columns:
            continue
        parents = DERIVED_PARENTS.get(col, [])
        if not parents:
            continue

        det_table: dict[tuple[str, ...], Any] = {}
        cond_counts: dict[tuple[str, ...], dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        present_counts: dict[tuple[str, ...], int] = defaultdict(int)
        missing_counts: dict[tuple[str, ...], int] = defaultdict(int)
        value_counts: dict[Any, int] = defaultdict(int)
        conflicts = 0

        for _, row in df.iterrows():
            parent_vals = tuple(_key(row[p]) for p in parents if p in df.columns)
            if any(v == "" for v in parent_vals):
                continue
            if any(pd.isna(row[p]) for p in parents if p in df.columns):
                continue

            present_counts[parent_vals] += 1
            if pd.isna(row[col]):
                missing_counts[parent_vals] += 1
                continue

            val = _to_int_or_val(row[col])
            value_counts[val] += 1
            if parent_vals in det_table and det_table[parent_vals] != val:
                conflicts += 1
            det_table[parent_vals] = val
            cond_counts[parent_vals][val] += 1

        total_present = sum(present_counts.values())
        if total_present:
            rules.missing_prob[col] = {
                key: missing_counts[key] / present_counts[key]
                for key in present_counts
            }
            rules.global_missing_prob[col] = sum(missing_counts.values()) / total_present
        if value_counts:
            rules.value_marginal[col] = _counts_to_probs(value_counts)

        partial_rows = 0
        partial_filled = 0
        for _, row in df.iterrows():
            parent_states = [pd.isna(row[p]) for p in parents if p in df.columns]
            if not parent_states or all(parent_states):
                continue
            if any(parent_states) and not all(parent_states):
                partial_rows += 1
                if not pd.isna(row[col]):
                    partial_filled += 1
        if partial_rows:
            rules.partial_parent_fill_rate[col] = partial_filled / partial_rows

        if conflicts == 0 and det_table:
            rules.deterministic[col] = det_table
        else:
            rules.conditional[col] = _smooth_probs(cond_counts)

        if col in ("qnowtg", "qnobeseg", "qnunwtg"):
            rules.bmi_fallback[col] = _learn_bmi_fallback(df, col)

    return rules


def apply_global_indicators(
    df: pd.DataFrame,
    rules: DerivedRules,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Calcula variables derivadas en el DataFrame."""
    out = df.copy()

    for col in DERIVED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

        parents = DERIVED_PARENTS.get(col, [])
        for idx, row in out.iterrows():
            parent_missing = [
                p not in out.columns or pd.isna(row[p]) for p in parents
            ]
            if any(parent_missing):
                fill_rate = rules.partial_parent_fill_rate.get(col, 0.0)
                marginal = rules.value_marginal.get(col)
                if fill_rate <= 0 or marginal is None or rng.random() >= fill_rate:
                    out.at[idx, col] = pd.NA
                    continue
                out.at[idx, col] = _sample_from_dist(marginal, rng)
                continue

            parent_key = tuple(_key(row[p]) for p in parents)

            p_miss = rules.missing_prob.get(col, {}).get(
                parent_key,
                rules.global_missing_prob.get(col, 0.0),
            )
            if rng.random() < p_miss:
                out.at[idx, col] = pd.NA
                continue

            if col in rules.deterministic:
                table = rules.deterministic[col]
                if parent_key in table:
                    out.at[idx, col] = table[parent_key]
                    continue
                marginal = rules.value_marginal.get(col)
                if marginal:
                    out.at[idx, col] = _sample_from_dist(marginal, rng)
                    continue

            if col in rules.bmi_fallback and col in ("qnowtg", "qnobeseg", "qnunwtg"):
                bmi_val = _bmi_from_row(row)
                if bmi_val is not None:
                    computed = _bmi_category(col, bmi_val)
                    fallback_key = (_key(row["Q1"]), _key(row["Q4"]), _key(row["Q5"]))
                    fb = rules.bmi_fallback[col]
                    out.at[idx, col] = fb.get(fallback_key, computed)
                    continue

            if col in rules.conditional:
                dist = rules.conditional[col].get(parent_key)
                if dist:
                    out.at[idx, col] = _sample_from_dist(dist, rng)
                else:
                    out.at[idx, col] = _sample_marginal(rules.conditional[col], rng)
                continue

            marginal = rules.value_marginal.get(col)
            if marginal:
                out.at[idx, col] = _sample_from_dist(marginal, rng)
            else:
                out.at[idx, col] = pd.NA

    return out


def _learn_bmi_fallback(df: pd.DataFrame, col: str) -> dict[tuple[str, ...], Any]:
    table: dict[tuple[str, ...], Any] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get(col)) or pd.isna(row.get("Q4")) or pd.isna(row.get("Q5")):
            continue
        key = (_key(row["Q1"]), _key(row["Q4"]), _key(row["Q5"]))
        table[key] = _to_int_or_val(row[col])
    return table


def _bmi_from_row(row: pd.Series) -> float | None:
    try:
        h = float(row["Q4"])
        w = float(row["Q5"])
        if h <= 0:
            return None
        return w / (h * h)
    except (TypeError, ValueError, KeyError):
        return None


def _bmi_category(col: str, bmi: float) -> int:
    """Categorías binarias estándar WHO (1=condición, 2=no)."""
    if col == "qnobeseg":
        return 1 if bmi >= 30 else 2
    if col == "qnunwtg":
        return 1 if bmi < 18.5 else 2
    if col == "qnowtg":
        return 1 if bmi >= 25 else 2
    return 2


def _counts_to_probs(counts: dict[Any, int], alpha: float = 1.0) -> dict[Any, float]:
    total = sum(counts.values()) + alpha * len(counts)
    return {value: (count + alpha) / total for value, count in counts.items()}


def _smooth_probs(
    counts: dict[tuple[str, ...], dict[Any, int]],
    alpha: float = 1.0,
) -> dict[tuple[str, ...], dict[Any, float]]:
    result: dict[tuple[str, ...], dict[Any, float]] = {}
    for key, val_counts in counts.items():
        total = sum(val_counts.values()) + alpha * len(val_counts)
        result[key] = {v: (c + alpha) / total for v, c in val_counts.items()}
    return result


def _sample_from_dist(dist: dict[Any, float], rng: np.random.Generator) -> Any:
    values = list(dist.keys())
    probs = np.array([dist[v] for v in values], dtype=float)
    probs /= probs.sum()
    choice = rng.choice(values, p=probs)
    return _to_int_or_val(choice)


def _sample_marginal(
    all_dists: dict[tuple[str, ...], dict[Any, float]],
    rng: np.random.Generator,
) -> Any:
    if not all_dists:
        return pd.NA
    keys = list(all_dists.keys())
    pick = keys[int(rng.integers(0, len(keys)))]
    return _sample_from_dist(all_dists[pick], rng)


def _key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _to_int_or_val(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float) and value == int(value):
        return int(value)
    try:
        f = float(value)
        if f == int(f):
            return int(f)
        return f
    except (TypeError, ValueError):
        return value
