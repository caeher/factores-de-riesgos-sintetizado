"""Constantes y utilidades para la encuesta SLV2013."""

from __future__ import annotations

import math
from typing import Any

SPSS_MISSING = 1.7976931348623157e308
SPSS_MISSING_STR = "1.79769313486232e+308"

# Columnas presentes en el CSV (orden del header original).
ALL_COLUMNS: list[str] = [
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
    "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20",
    "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q34", "Q35", "Q36",
    "Q37", "Q38", "Q39", "Q40", "Q44", "Q45", "Q46", "Q47", "Q48", "Q49",
    "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q56", "Q57", "Q58",
    "QN6", "QN7", "QN8", "QN9", "QN10", "QN11", "QN12", "QN13", "QN14",
    "QN15", "QN16", "QN17", "QN18", "QN19", "QN20", "QN21", "QN22", "QN23",
    "QN24", "QN25", "QN26", "QN27", "QN34", "QN35", "QN36", "QN37", "QN38",
    "QN39", "QN40", "QN44", "QN45", "QN46", "QN47", "QN48", "QN49", "QN50",
    "QN51", "QN52", "QN53", "QN54", "QN55", "QN56", "QN57", "QN58",
    "qnowtg", "qnobeseg", "qnunwtg", "qnfrvgg", "qnpa7g", "qnpe5g", "qnc1g", "qnc2g",
    "weight", "stratum", "psu",
]

Q_COLUMNS: list[str] = [c for c in ALL_COLUMNS if c.startswith("Q") and not c.startswith("QN")]
QN_COLUMNS: list[str] = [c for c in ALL_COLUMNS if c.startswith("QN")]
DERIVED_COLUMNS: list[str] = [
    "qnowtg", "qnobeseg", "qnunwtg", "qnfrvgg", "qnpa7g", "qnpe5g", "qnc1g", "qnc2g",
]
DESIGN_COLUMNS: list[str] = ["weight", "stratum", "psu"]

# Columnas que entrenan el modelo SDV (capa A).
SYNTH_COLUMNS: list[str] = Q_COLUMNS + DESIGN_COLUMNS

CONTINUOUS_COLUMNS: set[str] = {"Q4", "Q5", "weight"}
INTEGER_COLUMNS: set[str] = {"Q5"}

QN_PARENT_MAP: dict[str, str] = {
    qn: f"Q{qn[2:]}" for qn in QN_COLUMNS
}

# Variables derivadas y sus columnas padre para reglas probabilísticas.
DERIVED_PARENTS: dict[str, list[str]] = {
    "qnowtg": ["Q1", "Q4", "Q5"],
    "qnobeseg": ["Q1", "Q4", "Q5"],
    "qnunwtg": ["Q1", "Q4", "Q5"],
    "qnfrvgg": ["Q34", "Q35", "Q36", "Q37"],
    "qnpa7g": ["Q44", "Q45", "Q46", "Q47", "Q48", "Q49", "Q50"],
    "qnpe5g": ["Q51", "Q52", "Q53", "Q54", "Q55", "Q56", "Q57", "Q58"],
    "qnc1g": ["Q6", "Q7", "Q8", "Q9", "Q10"],
    "qnc2g": ["Q11", "Q12", "Q13", "Q14", "Q15"],
}


def is_spss_missing(value: Any) -> bool:
    """True si el valor representa missing SPSS/Stata."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == SPSS_MISSING_STR:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        pass
    try:
        f = float(value)
        if not math.isfinite(f):
            return True
        return f >= SPSS_MISSING * 0.999
    except (TypeError, ValueError):
        return True


def qn_column_for(q_column: str) -> str | None:
    """Devuelve la columna QN asociada a una Q (solo Q6+)."""
    if not q_column.startswith("Q") or q_column.startswith("QN"):
        return None
    num = q_column[1:]
    if not num.isdigit() or int(num) < 6:
        return None
    qn = f"QN{num}"
    return qn if qn in QN_COLUMNS else None
