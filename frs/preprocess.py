"""Perfilado y patrones de missing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frs.constants import (
    ALL_COLUMNS,
    CONTINUOUS_COLUMNS,
    DESIGN_COLUMNS,
    Q_COLUMNS,
    SYNTH_COLUMNS,
)
from frs.io import infer_csv_export_format


@dataclass
class MissingPatterns:
    """Máscaras de missing agrupadas por estrato."""

    by_stratum: dict[str, list[dict[str, bool]]] = field(default_factory=dict)
    global_masks: list[dict[str, bool]] = field(default_factory=list)

    def sample_mask(
        self,
        stratum: Any,
        rng: np.random.Generator,
    ) -> dict[str, bool]:
        key = _str_key(stratum)
        pool = self.by_stratum.get(key) or self.global_masks
        if not pool:
            return {col: False for col in ALL_COLUMNS}
        idx = rng.integers(0, len(pool))
        return pool[idx]


def _str_key(stratum: Any) -> str:
    if pd.isna(stratum):
        return "__missing__"
    if isinstance(stratum, float) and stratum == int(stratum):
        return str(int(stratum))
    return str(stratum)


def _is_missing(series: pd.Series) -> pd.Series:
    return series.isna()


def infer_column_profile(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Infiere metadatos de una columna para schema.yaml."""
    series = df[column]
    missing_pct = float(_is_missing(series).mean() * 100)
    non_missing = series.dropna()

    if column in CONTINUOUS_COLUMNS:
        if len(non_missing) == 0:
            return {
                "dtype": "numerical",
                "missing_pct": round(missing_pct, 2),
            }
        vals = pd.to_numeric(non_missing, errors="coerce").dropna()
        return {
            "dtype": "numerical",
            "min": float(vals.min()),
            "max": float(vals.max()),
            "missing_pct": round(missing_pct, 2),
        }

    if column in DESIGN_COLUMNS and column == "weight":
        vals = pd.to_numeric(non_missing, errors="coerce").dropna()
        return {
            "dtype": "numerical",
            "min": float(vals.min()),
            "max": float(vals.max()),
            "missing_pct": round(missing_pct, 2),
        }

    categories = sorted({_normalize_cat(v) for v in non_missing})
    return {
        "dtype": "categorical",
        "categories": categories,
        "missing_pct": round(missing_pct, 2),
    }


def profile_dataframe(
    df: pd.DataFrame,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Genera schema completo desde DataFrame."""
    columns: dict[str, Any] = {}
    export_format: dict[str, dict[str, Any]] = {}
    if source_path is not None and Path(source_path).exists():
        export_format = infer_csv_export_format(source_path)

    for col in ALL_COLUMNS:
        if col in df.columns:
            profile = infer_column_profile(df, col)
            if col in export_format:
                profile.update(export_format[col])
            columns[col] = profile
    return {
        "dataset": "SLV2013_Public_Use",
        "n_rows": len(df),
        "synth_columns": SYNTH_COLUMNS,
        "columns": columns,
    }


def _normalize_cat(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def learn_missing_patterns(df: pd.DataFrame) -> MissingPatterns:
    """Aprende vectores de missing por estrato desde datos reales."""
    patterns = MissingPatterns()
    global_masks: list[dict[str, bool]] = []

    for _, row in df.iterrows():
        mask = {col: bool(pd.isna(row[col])) for col in ALL_COLUMNS if col in df.columns}
        global_masks.append(mask)

    patterns.global_masks = global_masks

    if "stratum" not in df.columns:
        return patterns

    for stratum, group in df.groupby("stratum", dropna=False):
        key = _str_key(stratum)
        patterns.by_stratum[key] = [
            {col: bool(pd.isna(row[col])) for col in ALL_COLUMNS if col in df.columns}
            for _, row in group.iterrows()
        ]

    return patterns


def apply_missing_mask(df: pd.DataFrame, mask: dict[str, bool]) -> pd.DataFrame:
    """Aplica una máscara de missing a un DataFrame (una fila o varias)."""
    out = df.copy()
    for col, is_miss in mask.items():
        if col in out.columns and is_miss:
            out[col] = pd.NA
    return out


def fill_sdv_missing(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sustituye NaN generados por SDV antes de aplicar máscaras de missing reales."""
    out = df.copy()
    for col in SYNTH_COLUMNS:
        if col not in out.columns:
            continue
        miss = out[col].isna()
        if not miss.any():
            continue
        observed = source_df[col].dropna()
        if observed.empty:
            continue
        n = int(miss.sum())
        repl = observed.sample(
            n,
            replace=True,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).to_numpy()
        out.loc[miss, col] = repl
    return out


def apply_missing_masks_by_stratum(
    df: pd.DataFrame,
    patterns: MissingPatterns,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Aplica máscaras de missing muestreadas por fila según estrato."""
    rows: list[pd.Series] = []
    for _, row in df.iterrows():
        mask = patterns.sample_mask(row.get("stratum"), rng)
        masked = apply_missing_mask(pd.DataFrame([row]), mask).iloc[0]
        rows.append(masked)
    return pd.DataFrame(rows, columns=df.columns)


def enforce_missing_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Reglas duras de co-ocurrencia de missing."""
    out = df.copy()

    # Si Q4 falta, derivadas de peso también.
    if "Q4" in out.columns:
        q4_miss = out["Q4"].isna()
        for col in ("Q5", "qnowtg", "qnobeseg", "qnunwtg"):
            if col in out.columns:
                out.loc[q4_miss, col] = pd.NA

    # Si Q falta, QN también debe faltar. No al revés: códigos como Q18=1
    # son válidos sin recodificación QN en los datos reales.
    from frs.constants import QN_PARENT_MAP

    for qn, q in QN_PARENT_MAP.items():
        if q in out.columns and qn in out.columns:
            q_miss = out[q].isna()
            out.loc[q_miss, qn] = pd.NA

    return out


def clip_and_round_synth(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Recorta y redondea valores sintéticos según schema."""
    out = df.copy()
    col_specs = schema.get("columns", {})

    if "Q4" in out.columns:
        out["Q4"] = pd.to_numeric(out["Q4"], errors="coerce").round(2)

    if "Q5" in out.columns:
        out["Q5"] = pd.to_numeric(out["Q5"], errors="coerce").round(0)

    for col in ("stratum", "psu"):
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = vals.apply(
                lambda v: int(v) if pd.notna(v) and float(v) == int(float(v)) else v
            )

    for col, spec in col_specs.items():
        if col not in out.columns:
            continue
        if spec.get("dtype") == "numerical":
            vals = pd.to_numeric(out[col], errors="coerce")
            if "min" in spec:
                vals = vals.clip(lower=spec["min"])
            if "max" in spec:
                vals = vals.clip(upper=spec["max"])
            out[col] = vals
        elif spec.get("dtype") == "categorical" and spec.get("categories"):
            allowed = {_normalize_cat(c) for c in spec["categories"]}
            allowed_numeric = []
            for c in allowed:
                try:
                    allowed_numeric.append(int(float(c)))
                except ValueError:
                    allowed_numeric.append(c)

            def _fix_cat(val: Any) -> Any:
                if pd.isna(val):
                    return pd.NA
                norm = _normalize_cat(val)
                if norm in allowed:
                    return int(float(norm)) if norm.isdigit() or _is_numeric(norm) else norm
                # Mapear al valor permitido más cercano numéricamente.
                try:
                    fval = float(val)
                    nums = [float(c) for c in allowed if _is_numeric(c)]
                    if nums:
                        closest = min(nums, key=lambda x: abs(x - fval))
                        return int(closest) if closest == int(closest) else closest
                except (TypeError, ValueError):
                    pass
                return allowed_numeric[0] if allowed_numeric else val

            out[col] = out[col].apply(_fix_cat)

    return out


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_valid_design_combos(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Catálogo de combinaciones (stratum, psu) observadas."""
    combos: set[tuple[str, str]] = set()
    for _, row in df.dropna(subset=["stratum", "psu"]).iterrows():
        combos.add((_str_key(row["stratum"]), _str_key(row["psu"])))
    return combos


def enforce_design_combos(
    df: pd.DataFrame,
    valid_combos: set[tuple[str, str]],
    source_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Reemplaza combos (stratum, psu) inválidos por uno observado."""
    if not valid_combos or "stratum" not in df.columns or "psu" not in df.columns:
        return df

    out = df.copy()
    combo_list = list(valid_combos)
    source = source_df.dropna(subset=["stratum", "psu"])

    for idx, row in out.iterrows():
        key = (_str_key(row["stratum"]), _str_key(row["psu"]))
        if key not in valid_combos:
            pick = source.sample(1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            out.at[idx, "stratum"] = pick["stratum"]
            out.at[idx, "psu"] = pick["psu"]
            if "weight" in out.columns and "weight" in pick.index:
                out.at[idx, "weight"] = pick["weight"]

    return out
