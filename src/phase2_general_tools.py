"""Fase 2: herramientas generales, graficos y estadisticos descriptivos."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from plotting import write_line_svg, write_two_panel_svg


INPUT_COLUMNS_AS_FLOAT = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_buy_quote",
    "log_close",
    "r",
    "r2",
    "abs_r",
    "hl_range",
    "log_volume",
    "log_trades",
    "rv_past_12",
    "rv_past_48",
    "rv_past_288",
    "rv_future_12",
    "rv_future_48",
    "log_rv_past_12",
    "log_rv_past_48",
    "log_rv_past_288",
    "log_rv_future_12",
    "log_rv_future_48",
}
INPUT_COLUMNS_AS_INT = {"trades"}

DESCRIPTIVE_COLUMNS = [
    "r",
    "abs_r",
    "r2",
    "log_rv_past_12",
    "log_rv_future_12",
    "volume",
    "trades",
]

FIGURE_SPECS = [
    {
        "file": "phase2_01_close.svg",
        "title": "BTCUSDT close - velas de 5 minutos",
        "y_label": "close",
        "column": "close",
        "color": "#1f6fb2",
    },
    {
        "file": "phase2_02_log_close.svg",
        "title": "Log precio BTCUSDT",
        "y_label": "log(close)",
        "column": "log_close",
        "color": "#3b7f5f",
    },
    {
        "file": "phase2_03_returns.svg",
        "title": "Retornos logaritmicos de 5 minutos",
        "y_label": "r_t",
        "column": "r",
        "color": "#5f6b7a",
    },
    {
        "file": "phase2_04_abs_returns.svg",
        "title": "Retornos absolutos de 5 minutos",
        "y_label": "|r_t|",
        "column": "abs_r",
        "color": "#b45f06",
    },
    {
        "file": "phase2_05_log_rv_past_12.svg",
        "title": "Volatilidad realizada pasada de 1 hora",
        "y_label": "log_rv_past_12",
        "column": "log_rv_past_12",
        "color": "#8a4f9f",
    },
    {
        "file": "phase2_06_log_rv_future_12.svg",
        "title": "Volatilidad realizada futura de 1 hora",
        "y_label": "log_rv_future_12",
        "column": "log_rv_future_12",
        "color": "#a53d3d",
    },
]


def read_features_csv(path: Path) -> list[dict[str, Any]]:
    """Lee el CSV de Fase 1 con conversion simple de tipos."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {"open_time": row["open_time"]}
            for column, value in row.items():
                if column == "open_time":
                    continue
                if value == "":
                    parsed[column] = None
                elif column in INPUT_COLUMNS_AS_INT:
                    parsed[column] = int(value)
                elif column in INPUT_COLUMNS_AS_FLOAT:
                    parsed[column] = float(value)
                else:
                    parsed[column] = value
            rows.append(parsed)
    return rows


def percentile(sorted_values: list[float], probability: float) -> float:
    """Percentil con interpolacion lineal."""
    if not sorted_values:
        raise ValueError("No hay valores para calcular percentiles")
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def descriptive_statistics(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    """Calcula estadisticos descriptivos basicos y momentos estandarizados."""
    result: list[dict[str, Any]] = []
    for column in columns:
        values = [float(row[column]) for row in rows if row[column] is not None]
        sorted_values = sorted(values)
        n = len(values)
        mean = sum(values) / n
        centered = [value - mean for value in values]
        m2 = sum(value * value for value in centered) / n
        sample_std = math.sqrt(sum(value * value for value in centered) / (n - 1))
        if m2 > 0:
            m3 = sum(value**3 for value in centered) / n
            m4 = sum(value**4 for value in centered) / n
            skewness = m3 / (m2**1.5)
            kurtosis_excess = m4 / (m2 * m2) - 3.0
        else:
            skewness = 0.0
            kurtosis_excess = 0.0

        min_index, min_value = min(enumerate(values), key=lambda item: item[1])
        max_index, max_value = max(enumerate(values), key=lambda item: item[1])
        result.append(
            {
                "variable": column,
                "n": n,
                "mean": mean,
                "std": sample_std,
                "min": sorted_values[0],
                "p01": percentile(sorted_values, 0.01),
                "p05": percentile(sorted_values, 0.05),
                "p25": percentile(sorted_values, 0.25),
                "p50": percentile(sorted_values, 0.50),
                "p75": percentile(sorted_values, 0.75),
                "p95": percentile(sorted_values, 0.95),
                "p99": percentile(sorted_values, 0.99),
                "max": sorted_values[-1],
                "skewness": skewness,
                "kurtosis_excess": kurtosis_excess,
                "min_time": rows[min_index]["open_time"],
                "max_time": rows[max_index]["open_time"],
            }
        )
    return result


def top_extremes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extrae eventos extremos utiles para interpretar los graficos."""
    extremes: list[dict[str, Any]] = []
    specs = [
        ("r", "min", "retorno mas negativo"),
        ("r", "max", "retorno mas positivo"),
        ("abs_r", "max", "mayor retorno absoluto"),
        ("log_rv_past_12", "max", "mayor log volatilidad pasada 1h"),
        ("log_rv_future_12", "max", "mayor log volatilidad futura 1h"),
        ("volume", "max", "mayor volumen"),
        ("trades", "max", "mayor numero de trades"),
    ]
    for column, mode, label in specs:
        values = [(index, float(row[column])) for index, row in enumerate(rows)]
        index, value = (min(values, key=lambda item: item[1]) if mode == "min" else max(values, key=lambda item: item[1]))
        extremes.append(
            {
                "event": label,
                "variable": column,
                "time": rows[index]["open_time"],
                "value": value,
                "close": rows[index]["close"],
                "log_rv_past_12": rows[index]["log_rv_past_12"],
                "log_rv_future_12": rows[index]["log_rv_future_12"],
            }
        )
    return extremes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta la Fase 2 del TFG BTC.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_features_csv(args.input)
    if not rows:
        raise SystemExit("No hay filas en el dataset procesado")

    tables_dir = args.reports_dir / "tables"
    figures_dir = args.reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    times = [row["open_time"] for row in rows]
    for spec in FIGURE_SPECS:
        write_line_svg(
            figures_dir / spec["file"],
            times,
            [float(row[spec["column"]]) for row in rows],
            title=spec["title"],
            y_label=spec["y_label"],
            color=spec["color"],
        )

    write_two_panel_svg(
        figures_dir / "phase2_07_volume_trades.svg",
        times,
        [float(row["volume"]) for row in rows],
        [float(row["trades"]) for row in rows],
        title="Volumen y numero de trades",
        first_y_label="volume",
        second_y_label="trades",
    )

    descriptive_rows = descriptive_statistics(rows, DESCRIPTIVE_COLUMNS)
    extremes = top_extremes(rows)

    write_rows_csv(
        tables_dir / "phase2_descriptive_statistics.csv",
        descriptive_rows,
        [
            "variable",
            "n",
            "mean",
            "std",
            "min",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "max",
            "skewness",
            "kurtosis_excess",
            "min_time",
            "max_time",
        ],
    )
    write_rows_csv(
        tables_dir / "phase2_extreme_events.csv",
        extremes,
        ["event", "variable", "time", "value", "close", "log_rv_past_12", "log_rv_future_12"],
    )
    print("FASE 2 - HERRAMIENTAS GENERALES")
    print("=" * 72)
    print(f"Filas analizadas: {len(rows):,}")
    print(f"Figuras SVG: {figures_dir}")
    print(f"Tabla estadistica: {tables_dir / 'phase2_descriptive_statistics.csv'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
