"""Validación de fidelidad entre datos reales y sintéticos."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from frs.constants import ALL_COLUMNS, CONTINUOUS_COLUMNS, DERIVED_COLUMNS
from frs.mappings import extract_qn_mappings
from frs.derive import DerivedRules, learn_derived_rules


def validate_synthesis(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    derived_rules: DerivedRules | None = None,
) -> dict[str, Any]:
    """Compara real vs sintético y devuelve reporte estructurado."""
    if derived_rules is None:
        derived_rules = learn_derived_rules(real_df)

    qn_maps = extract_qn_mappings(real_df)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_real": len(real_df),
        "n_synthetic": len(synth_df),
        "ks_tests": {},
        "categorical_tvd": {},
        "missing_rate_diff_pp": {},
        "qn_coherence": {},
        "derived_coherence": {},
        "summary": {"passed": True, "warnings": []},
    }

    for col in CONTINUOUS_COLUMNS | {"weight"}:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        real_vals = pd.to_numeric(real_df[col], errors="coerce").dropna()
        synth_vals = pd.to_numeric(synth_df[col], errors="coerce").dropna()
        if len(real_vals) < 2 or len(synth_vals) < 2:
            continue
        stat, pval = stats.ks_2samp(real_vals, synth_vals)
        report["ks_tests"][col] = {
            "statistic": float(stat),
            "p_value": float(pval),
            "passed": bool(pval > 0.05 or stat < 0.1),
        }
        if not report["ks_tests"][col]["passed"]:
            report["summary"]["warnings"].append(f"KS test failed for {col}")

    for col in ALL_COLUMNS:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        if col in CONTINUOUS_COLUMNS or col == "weight":
            continue
        tvd = _total_variation_distance(real_df[col], synth_df[col])
        report["categorical_tvd"][col] = {
            "tvd": float(tvd),
            "passed": bool(tvd < 0.05),
        }
        if tvd >= 0.05:
            report["summary"]["warnings"].append(f"TVD high for {col}: {tvd:.3f}")

    for col in ALL_COLUMNS:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        real_miss = float(real_df[col].isna().mean() * 100)
        synth_miss = float(synth_df[col].isna().mean() * 100)
        diff = abs(real_miss - synth_miss)
        report["missing_rate_diff_pp"][col] = {
            "real_pct": round(real_miss, 2),
            "synth_pct": round(synth_miss, 2),
            "diff_pp": round(diff, 2),
            "passed": bool(diff < 5.0),
        }
        if diff >= 5.0:
            report["summary"]["warnings"].append(
                f"Missing rate diff for {col}: {diff:.1f} pp"
            )

    report["qn_coherence"] = _check_qn_coherence(synth_df, qn_maps)
    report["derived_coherence"] = _check_derived_coherence(synth_df, derived_rules)

    if report["summary"]["warnings"]:
        report["summary"]["passed"] = len(report["summary"]["warnings"]) <= 5

    return report


def save_validation_report(report: dict[str, Any], path: str | Path) -> None:
    """Guarda reporte JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(_report_to_markdown(report))


def _total_variation_distance(real: pd.Series, synth: pd.Series) -> float:
    real_dist = _value_distribution(real)
    synth_dist = _value_distribution(synth)
    all_keys = set(real_dist) | set(synth_dist)
    return 0.5 * sum(abs(real_dist.get(k, 0) - synth_dist.get(k, 0)) for k in all_keys)


def _normalize_dist_key(value: Any) -> str | None:
    """Normaliza valores para comparar distribuciones (1.0 == 1)."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    try:
        f = float(value)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (TypeError, ValueError):
        return str(value)


def _value_distribution(series: pd.Series) -> dict[str, float]:
    clean = series.dropna().map(_normalize_dist_key)
    if len(clean) == 0:
        return {}
    counts = clean.value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def _check_qn_coherence(
    df: pd.DataFrame,
    qn_maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from frs.constants import QN_PARENT_MAP

    total = 0
    matches = 0
    for qn_col, table in qn_maps.items():
        q_col = QN_PARENT_MAP[qn_col]
        if q_col not in df.columns or qn_col not in df.columns:
            continue
        for _, row in df.iterrows():
            if pd.isna(row[q_col]) and pd.isna(row[qn_col]):
                matches += 1
                total += 1
                continue
            if pd.isna(row[q_col]) or pd.isna(row[qn_col]):
                total += 1
                continue
            total += 1
            key = str(int(float(row[q_col]))) if _is_num(row[q_col]) else str(row[q_col])
            expected = table.get(key)
            actual = row[qn_col]
            if expected is not None and _values_equal(expected, actual):
                matches += 1

    rate = matches / total if total else 1.0
    return {"match_rate": round(rate, 4), "passed": rate >= 0.999}


def _check_derived_coherence(df: pd.DataFrame, rules: DerivedRules) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for col in DERIVED_COLUMNS:
        if col not in df.columns:
            continue
        if col in rules.deterministic:
            table = rules.deterministic[col]
            parents = list(rules.deterministic[col].keys())
            if not parents:
                continue
            from frs.constants import DERIVED_PARENTS

            parent_cols = DERIVED_PARENTS.get(col, [])
            total = 0
            matches = 0
            for _, row in df.iterrows():
                if pd.isna(row[col]):
                    continue
                if any(pd.isna(row[p]) for p in parent_cols if p in df.columns):
                    continue
                key = tuple(
                    str(int(float(row[p]))) if _is_num(row[p]) else str(row[p])
                    for p in parent_cols
                )
                total += 1
                if key in table and _values_equal(table[key], row[col]):
                    matches += 1
            rate = matches / total if total else 1.0
            results[col] = {"match_rate": round(rate, 4), "passed": rate >= 0.95}
    return results


def _is_num(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _values_equal(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return int(float(a)) == int(float(b))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reporte de validación — SLV2013",
        "",
        f"- Generado: {report['generated_at']}",
        f"- Real: {report['n_real']} filas",
        f"- Sintético: {report['n_synthetic']} filas",
        f"- Estado: {'OK' if report['summary']['passed'] else 'REVISAR'}",
        "",
    ]
    if report["summary"]["warnings"]:
        lines.append("## Advertencias")
        for w in report["summary"]["warnings"][:20]:
            lines.append(f"- {w}")
        lines.append("")

    if report.get("qn_coherence"):
        qc = report["qn_coherence"]
        lines.append(f"## Coherencia Q↔QN: {qc.get('match_rate', 0):.2%}")
        lines.append("")

    return "\n".join(lines)
