"""Fase 5: graficos de recurrencia iniciales."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from recurrence import recurrence_epsilon_for_rr, write_recurrence_png, zscore


SERIES_SPECS = [
    {"key": "r", "column": "r", "label": "Retornos logaritmicos r"},
    {"key": "abs_r", "column": "abs_r", "label": "Retornos absolutos |r|"},
    {
        "key": "log_rv_past_12",
        "column": "log_rv_past_12",
        "label": "v_t = log_rv_past_12",
    },
]

WINDOW_SIZE = 2000
TARGET_RR = 0.05
RAW_EXPECTED_ROWS = 245088


def read_features(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Lee fechas y series necesarias desde el dataset procesado."""
    times: list[str] = []
    series = {spec["key"]: [] for spec in SERIES_SPECS}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(row["open_time"])
            for spec in SERIES_SPECS:
                series[spec["key"]].append(float(row[spec["column"]]))
    return times, series


def select_windows(times: list[str], v: list[float], window_size: int) -> list[dict[str, Any]]:
    """Selecciona ventanas representativas de forma reproducible."""
    n = len(v)
    if n < window_size:
        raise ValueError("El dataset es menor que la ventana solicitada")

    rolling_means = rolling_window_means(v, window_size)
    quiet_target = percentile(sorted(rolling_means), 0.10)
    quiet_start = min(
        range(len(rolling_means)),
        key=lambda index: abs(rolling_means[index] - quiet_target),
    )

    high_center = max(range(n), key=lambda index: v[index])
    high_start = clamp(high_center - window_size // 2, 0, n - window_size)

    recent_start = n - window_size
    middle_start = n // 2 - window_size // 2

    windows = [
        {
            "window": "quiet",
            "description": "Ventana tranquila",
            "selection_method": (
                "rolling mean de log_rv_past_12 mas cercano al percentil 10 "
                f"de medias rolling de {window_size} observaciones"
            ),
            "start_index": quiet_start,
            "reference_value": quiet_target,
        },
        {
            "window": "high_volatility",
            "description": "Ventana de alta volatilidad",
            "selection_method": (
                "ventana centrada en el maximo de log_rv_past_12 dentro del dataset procesado"
            ),
            "start_index": high_start,
            "reference_value": v[high_center],
            "reference_time": times[high_center],
        },
        {
            "window": "recent",
            "description": "Ventana reciente",
            "selection_method": "ultimas observaciones disponibles del dataset procesado",
            "start_index": recent_start,
            "reference_value": "",
        },
        {
            "window": "middle",
            "description": "Ventana continua representativa",
            "selection_method": "bloque continuo centrado en la mitad del dataset procesado",
            "start_index": middle_start,
            "reference_value": "",
        },
    ]

    for window in windows:
        start = window["start_index"]
        end_exclusive = start + window_size
        window["end_index_exclusive"] = end_exclusive
        window["start_time"] = times[start]
        window["end_time"] = times[end_exclusive - 1]
        window["n"] = window_size
        window["mean_log_rv_past_12"] = sum(v[start:end_exclusive]) / window_size
        window["min_log_rv_past_12"] = min(v[start:end_exclusive])
        window["max_log_rv_past_12"] = max(v[start:end_exclusive])

    return windows


def rolling_window_means(values: list[float], window_size: int) -> list[float]:
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return [
        (prefix[index + window_size] - prefix[index]) / window_size
        for index in range(0, len(values) - window_size + 1)
    ]


def percentile(sorted_values: list[float], probability: float) -> float:
    if probability <= 0.0:
        return sorted_values[0]
    if probability >= 1.0:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}"
        else:
            rounded[key] = value
    return rounded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 5: recurrence plots iniciales.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--target-rr", type=float, default=TARGET_RR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    times, series = read_features(args.input)
    n = len(times)
    if args.window_size != WINDOW_SIZE or abs(args.target_rr - TARGET_RR) > 1e-12:
        raise SystemExit(
            "Esta fase esta fijada a window-size=2000 y target-rr=0.05 para "
            "mantener comparabilidad entre ventanas."
        )

    reports_dir = args.reports_dir
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    windows = select_windows(times, series["log_rv_past_12"], args.window_size)
    parameter_rows: list[dict[str, Any]] = []

    for window in windows:
        start = window["start_index"]
        end = window["end_index_exclusive"]
        print(f"Ventana {window['window']}: {window['start_time']} -> {window['end_time']}")
        for spec in SERIES_SPECS:
            key = spec["key"]
            raw_values = series[key][start:end]
            normalized_values, mean_before, std_before = zscore(raw_values)
            epsilon, estimated_rr = recurrence_epsilon_for_rr(normalized_values, args.target_rr)
            figure_file = f"phase5_{window['window']}_{key}_rp.png"
            achieved_rr = write_recurrence_png(figures_dir / figure_file, normalized_values, epsilon)
            parameter_rows.append(
                {
                    "series": key,
                    "series_label": spec["label"],
                    "window": window["window"],
                    "window_description": window["description"],
                    "selection_method": window["selection_method"],
                    "processed_dataset_rows": n,
                    "window_start_index": start,
                    "window_end_index_exclusive": end,
                    "start_time": window["start_time"],
                    "end_time": window["end_time"],
                    "n": args.window_size,
                    "normalization": "z-score por ventana",
                    "metric": "absolute_distance_1d",
                    "target_rr": args.target_rr,
                    "epsilon": epsilon,
                    "estimated_rr_from_sorted_values": estimated_rr,
                    "achieved_rr": achieved_rr,
                    "mean_before_zscore": mean_before,
                    "std_before_zscore": std_before,
                    "min_zscore": min(normalized_values),
                    "max_zscore": max(normalized_values),
                    "figure_file": figure_file,
                }
            )

    write_rows_csv(
        tables_dir / "phase5_recurrence_parameters.csv",
        parameter_rows,
        [
            "series",
            "series_label",
            "window",
            "window_description",
            "selection_method",
            "processed_dataset_rows",
            "window_start_index",
            "window_end_index_exclusive",
            "start_time",
            "end_time",
            "n",
            "normalization",
            "metric",
            "target_rr",
            "epsilon",
            "estimated_rr_from_sorted_values",
            "achieved_rr",
            "mean_before_zscore",
            "std_before_zscore",
            "min_zscore",
            "max_zscore",
            "figure_file",
        ],
    )
    write_rows_csv(
        tables_dir / "phase5_selected_windows.csv",
        windows,
        [
            "window",
            "description",
            "selection_method",
            "start_index",
            "end_index_exclusive",
            "start_time",
            "end_time",
            "n",
            "reference_value",
            "reference_time",
            "mean_log_rv_past_12",
            "min_log_rv_past_12",
            "max_log_rv_past_12",
        ],
    )


    print("FASE 5 - GRAFICOS DE RECURRENCIA INICIALES")
    print("=" * 72)
    print(f"Dataset procesado: {n:,} observaciones")
    print(f"Ventanas: {len(windows)} x series: {len(SERIES_SPECS)}")
    print(f"Figuras: {figures_dir}")
    print(f"Tabla parametros: {tables_dir / 'phase5_recurrence_parameters.csv'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
