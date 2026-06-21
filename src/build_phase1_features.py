"""Fase 1: construccion de variables y volatilidades realizadas."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data_loading import read_klines_csv, write_rows_csv
from preprocessing import (
    CREATED_COLUMNS,
    FEATURE_DEFINITIONS,
    FINAL_COLUMNS,
    Phase1Result,
    build_phase1_features,
)


DEFAULT_INPUT = Path("btc_5m_clean.csv")
DEFAULT_OUTPUT = Path("data/processed/btc_5m_features.csv")
DEFAULT_REPORTS_DIR = Path("reports")


def write_simple_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_rows_csv(path, rows, columns)


def build_shape_summary(result: Phase1Result) -> list[dict[str, Any]]:
    return [
        {"metric": "input_rows", "value": result.input_rows},
        {"metric": "output_rows_after_dropna", "value": result.output_rows},
        {"metric": "rows_dropped_total", "value": result.input_rows - result.output_rows},
        {"metric": "rows_dropped_start", "value": result.rows_dropped_start},
        {"metric": "rows_dropped_end", "value": result.rows_dropped_end},
        {"metric": "first_valid_index_zero_based", "value": result.first_valid_index},
        {"metric": "last_valid_index_zero_based", "value": result.last_valid_index},
        {"metric": "final_start_open_time", "value": result.rows[0]["open_time"]},
        {"metric": "final_end_open_time", "value": result.rows[-1]["open_time"]},
        {"metric": "epsilon_for_log_rv", "value": result.epsilon},
    ]


def build_created_columns_table() -> list[dict[str, Any]]:
    return [
        {"column": column, "definition": FEATURE_DEFINITIONS[column]}
        for column in CREATED_COLUMNS
    ]


def build_feature_range_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for column in CREATED_COLUMNS:
        values = [row[column] for row in rows if row[column] is not None]
        table.append(
            {
                "column": column,
                "missing_after_dropna": len(rows) - len(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return table


def build_zero_rv_table(result: Phase1Result) -> list[dict[str, Any]]:
    return [
        {
            "column": column,
            "zero_or_negative_windows_before_dropna": count,
            "epsilon_used_for_log": result.epsilon,
        }
        for column, count in result.zero_rv_counts.items()
    ]


def write_phase1_tables(
    result: Phase1Result,
    reports_dir: Path,
) -> None:
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    shape_rows = build_shape_summary(result)
    created_rows = build_created_columns_table()
    range_rows = build_feature_range_table(result.rows)
    zero_rows = build_zero_rv_table(result)
    head_rows = result.rows[:5]
    tail_rows = result.rows[-5:]

    write_simple_csv(tables_dir / "phase1_shape_summary.csv", shape_rows, ["metric", "value"])
    write_simple_csv(
        tables_dir / "phase1_created_columns.csv",
        created_rows,
        ["column", "definition"],
    )
    write_simple_csv(
        tables_dir / "phase1_feature_ranges.csv",
        range_rows,
        ["column", "missing_after_dropna", "min", "max"],
    )
    write_simple_csv(
        tables_dir / "phase1_zero_rv_windows.csv",
        zero_rows,
        ["column", "zero_or_negative_windows_before_dropna", "epsilon_used_for_log"],
    )
    write_simple_csv(tables_dir / "phase1_head.csv", head_rows, FINAL_COLUMNS)
    write_simple_csv(tables_dir / "phase1_tail.csv", tail_rows, FINAL_COLUMNS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construye variables de Fase 1.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_klines_csv(args.input)
    result = build_phase1_features(rows, epsilon=args.epsilon)

    write_rows_csv(args.output, result.rows, FINAL_COLUMNS)
    write_phase1_tables(result, args.reports_dir)

    print("FASE 1 - CONSTRUCCION DE VARIABLES")
    print("=" * 72)
    print(f"Entrada: {args.input}")
    print(f"Salida:  {args.output}")
    print(f"Shape entrada: ({result.input_rows:,}, 8)")
    print(f"Shape final:   ({result.output_rows:,}, {len(FINAL_COLUMNS)})")
    print(f"Rango final:   {result.rows[0]['open_time']} -> {result.rows[-1]['open_time']}")
    print(f"Filas eliminadas al inicio: {result.rows_dropped_start:,}")
    print(f"Filas eliminadas al final:  {result.rows_dropped_end:,}")
    print("Columnas creadas:")
    for column in CREATED_COLUMNS:
        print(f"- {column}: {FEATURE_DEFINITIONS[column]}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
