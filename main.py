"""CLI para síntesis de datos SLV2013."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from frs.io import load_schema, load_survey, save_schema
from frs.preprocess import profile_dataframe
from frs.synthesizer import SurveySynthesizer, synthesize
from frs.validate import save_validation_report, validate_synthesis


DEFAULT_INPUT = Path("data/input/SLV2013_Public_Use.csv")
DEFAULT_SCHEMA = Path("config/schema.yaml")
DEFAULT_OUTPUT_DIR = Path("data/output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FRS — Síntesis de datos de encuesta SLV2013",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    profile_cmd = sub.add_parser("profile", help="Generar schema.yaml desde CSV")
    profile_cmd.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT, help="CSV de entrada",
    )
    profile_cmd.add_argument(
        "--schema", "-s", type=Path, default=DEFAULT_SCHEMA, help="Ruta schema.yaml",
    )

    synth_cmd = sub.add_parser("synthesize", help="Generar datos sintéticos")
    synth_cmd.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT, help="CSV de entrada",
    )
    synth_cmd.add_argument(
        "--output", "-o", type=Path, default=None, help="CSV de salida",
    )
    synth_cmd.add_argument(
        "--rows", "-n", type=int, required=True, help="Número de filas sintéticas",
    )
    synth_cmd.add_argument(
        "--seed", type=int, default=None, help="Semilla aleatoria",
    )
    synth_cmd.add_argument(
        "--schema", "-s", type=Path, default=DEFAULT_SCHEMA, help="Ruta schema.yaml",
    )
    synth_cmd.add_argument(
        "--validate", action="store_true", help="Generar reporte de validación",
    )

    return parser


def cmd_profile(args: argparse.Namespace) -> None:
    df = load_survey(args.input)
    schema = profile_dataframe(df)
    save_schema(schema, args.schema)
    print(f"Schema guardado en {args.schema} ({len(schema['columns'])} columnas)")


def cmd_synthesize(args: argparse.Namespace) -> None:
    output = args.output
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = DEFAULT_OUTPUT_DIR / f"synthetic_SLV2013_{ts}.csv"

    real_df = load_survey(args.input)
    synth = SurveySynthesizer(schema_path=args.schema)

    schema = load_schema(args.schema)
    if not schema.get("columns"):
        synth.profile_and_save_schema(real_df)
        synth.schema = load_schema(args.schema)
    else:
        synth.schema = schema

    synth.fit(real_df)
    result = synth.sample(n=args.rows, seed=args.seed)

    from frs.io import save_survey

    save_survey(result, output)
    print(f"Generadas {len(result)} filas -> {output}")

    if args.validate:
        report = validate_synthesis(real_df, result, synth._derived_rules)
        report_path = output.with_suffix(".validation.json")
        save_validation_report(report, report_path)
        status = "OK" if report["summary"]["passed"] else "REVISAR"
        print(f"Validacion: {status} -> {report_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "profile":
        cmd_profile(args)
    elif args.command == "synthesize":
        cmd_synthesize(args)


if __name__ == "__main__":
    main()
