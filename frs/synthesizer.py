"""Motor de síntesis de datos SLV2013."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer

from frs.constants import ALL_COLUMNS, SYNTH_COLUMNS
from frs.derive import DerivedRules, apply_global_indicators, learn_derived_rules
from frs.io import load_schema, load_survey, save_survey
from frs.mappings import apply_qn_recodes, extract_qn_mappings
from frs.metadata import build_sdv_metadata, prepare_fit_dataframe, prepare_sample_dataframe
from frs.preprocess import (
    MissingPatterns,
    apply_missing_masks_by_stratum,
    clip_and_round_synth,
    enforce_design_combos,
    enforce_missing_rules,
    fill_sdv_missing,
    get_valid_design_combos,
    learn_missing_patterns,
    profile_dataframe,
)
from frs.io import save_schema


class SurveySynthesizer:
    """Orquestador fit/sample para encuesta SLV2013."""

    def __init__(self, schema_path: str | Path = "config/schema.yaml") -> None:
        self.schema_path = Path(schema_path)
        self.schema: dict[str, Any] = load_schema(self.schema_path)
        self._source_df: pd.DataFrame | None = None
        self._synthesizer: GaussianCopulaSynthesizer | None = None
        self._qn_mappings: dict[str, dict[str, Any]] = {}
        self._derived_rules: DerivedRules = DerivedRules()
        self._missing_patterns: MissingPatterns = MissingPatterns()
        self._valid_combos: set[tuple[str, str]] = set()

    def fit(self, df: pd.DataFrame) -> None:
        """Entrena modelo y aprende reglas auxiliares."""
        self._source_df = df.copy()
        self._qn_mappings = extract_qn_mappings(df)
        self._derived_rules = learn_derived_rules(df)
        self._missing_patterns = learn_missing_patterns(df)
        self._valid_combos = get_valid_design_combos(df)

        fit_df = prepare_fit_dataframe(df)
        metadata = build_sdv_metadata(df, self.schema)
        self._synthesizer = GaussianCopulaSynthesizer(metadata)
        self._synthesizer.fit(fit_df)

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        """Genera n filas sintéticas con derivaciones y missing SPSS-ready."""
        if self._synthesizer is None or self._source_df is None:
            raise RuntimeError("Debe llamar fit() antes de sample().")

        rng = np.random.default_rng(seed)
        if seed is not None:
            self._synthesizer._set_random_state(seed)

        raw = self._synthesizer.sample(num_rows=n)
        core = prepare_sample_dataframe(raw)

        core = clip_and_round_synth(core, self.schema)
        core = enforce_design_combos(core, self._valid_combos, self._source_df, rng)
        core = fill_sdv_missing(core, self._source_df, rng)
        core = apply_missing_masks_by_stratum(core, self._missing_patterns, rng)
        core = enforce_missing_rules(core)

        # Asegurar columnas Q completas antes de derivar.
        for col in SYNTH_COLUMNS:
            if col not in core.columns:
                core[col] = pd.NA

        full = core[SYNTH_COLUMNS].copy()
        for col in ALL_COLUMNS:
            if col not in full.columns:
                full[col] = pd.NA

        full = apply_qn_recodes(full, self._qn_mappings)
        full = apply_global_indicators(full, self._derived_rules, rng)
        full = enforce_missing_rules(full)

        return full.reindex(columns=ALL_COLUMNS)

    def profile_and_save_schema(
        self,
        df: pd.DataFrame,
        source_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Genera y persiste schema desde datos."""
        self.schema = profile_dataframe(df, source_path=source_path)
        save_schema(self.schema, self.schema_path)
        return self.schema


def synthesize(
    input_path: str | Path,
    output_path: str | Path,
    n: int,
    seed: int | None = None,
    schema_path: str | Path = "config/schema.yaml",
) -> pd.DataFrame:
    """Pipeline completo: cargar → fit → sample → guardar."""
    df = load_survey(input_path)
    synth = SurveySynthesizer(schema_path=schema_path)

    schema_file = Path(schema_path)
    if not schema_file.exists() or not load_schema(schema_path).get("columns"):
        synth.profile_and_save_schema(df, source_path=input_path)
        synth.schema = load_schema(schema_path)
    else:
        synth.schema = load_schema(schema_path)

    synth.fit(df)
    result = synth.sample(n=n, seed=seed)
    save_survey(result, output_path, schema=synth.schema, reference_path=input_path)
    return result
