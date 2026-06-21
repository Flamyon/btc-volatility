"""Fase 3: estacionariedad y transformaciones."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from plotting import write_line_svg, write_two_panel_svg
from stationarity import adf_select_lag, kpss_test, rolling_mean_std


SERIES_SPECS = [
    {
        "key": "close",
        "label": "A) close",
        "column": "close",
        "title": "Precio BTCUSDT close",
        "y_label": "close",
        "color": "#1f6fb2",
    },
    {
        "key": "log_close",
        "label": "B) log_close",
        "column": "log_close",
        "title": "Log precio BTCUSDT",
        "y_label": "log_close",
        "color": "#3b7f5f",
    },
    {
        "key": "r",
        "label": "C) r",
        "column": "r",
        "title": "Retornos logaritmicos",
        "y_label": "r",
        "color": "#5f6b7a",
    },
    {
        "key": "abs_r",
        "label": "D) abs_r",
        "column": "abs_r",
        "title": "Retornos absolutos",
        "y_label": "abs_r",
        "color": "#b45f06",
    },
    {
        "key": "log_rv_past_12",
        "label": "E) log_rv_past_12",
        "column": "log_rv_past_12",
        "title": "Log volatilidad realizada pasada 1h",
        "y_label": "log_rv_past_12",
        "color": "#8a4f9f",
    },
]

ROLLING_WINDOW = 288
ADF_LAG_CANDIDATES = [0, 1, 2, 3, 6, 12]


def read_features(path: Path) -> list[dict[str, Any]]:
    """Lee el dataset de Fase 1 con conversion numerica basica."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {"open_time": row["open_time"]}
            for column, value in row.items():
                if column == "open_time":
                    continue
                parsed[column] = float(value)
            rows.append(parsed)
    return rows


def build_rolling_summary(values: list[float], means: list[float], stds: list[float]) -> dict[str, Any]:
    """Resume estabilidad visual de media y varianza rolling."""
    first_half = values[: len(values) // 2]
    second_half = values[len(values) // 2 :]
    first_mean = sum(first_half) / len(first_half)
    second_mean = sum(second_half) / len(second_half)
    first_std = _sample_std(first_half)
    second_std = _sample_std(second_half)
    return {
        "raw_mean_first_half": first_mean,
        "raw_mean_second_half": second_mean,
        "raw_mean_difference": second_mean - first_mean,
        "raw_std_first_half": first_std,
        "raw_std_second_half": second_std,
        "raw_std_ratio_second_over_first": second_std / first_std if first_std > 0 else "",
        "rolling_mean_min": min(means),
        "rolling_mean_max": max(means),
        "rolling_std_min": min(stds),
        "rolling_std_max": max(stds),
    }


def _rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}"
        else:
            rounded[key] = value
    return rounded


def _sample_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta la Fase 3 de estacionariedad.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_features(args.input)
    if not rows:
        raise SystemExit("No hay filas en el dataset procesado")

    times = [row["open_time"] for row in rows]
    reports_dir = args.reports_dir
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adf_rows: list[dict[str, Any]] = []
    adf_lag_rows: list[dict[str, Any]] = []
    kpss_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []

    for spec in SERIES_SPECS:
        key = spec["key"]
        values = [float(row[spec["column"]]) for row in rows]

        write_line_svg(
            figures_dir / f"phase3_{key}_time.svg",
            times,
            values,
            title=f"{spec['label']} - {spec['title']}",
            y_label=spec["y_label"],
            color=spec["color"],
        )

        rolling_means, rolling_stds = rolling_mean_std(values, ROLLING_WINDOW)
        rolling_times = times[ROLLING_WINDOW - 1 :]
        write_two_panel_svg(
            figures_dir / f"phase3_{key}_rolling_mean_std.svg",
            rolling_times,
            rolling_means,
            rolling_stds,
            title=f"{spec['label']} - media y desviacion rolling 1 dia",
            first_y_label="media rolling",
            second_y_label="std rolling",
            first_color=spec["color"],
            second_color="#7a4f2a",
        )

        adf_selected, adf_candidates = adf_select_lag(values, ADF_LAG_CANDIDATES)
        adf_rows.append(
            {
                "series": key,
                "regression": "constant",
                "selected_lag": adf_selected.lag,
                "nobs": adf_selected.nobs,
                "adf_statistic": adf_selected.statistic,
                "gamma": adf_selected.gamma,
                "std_error": adf_selected.std_error,
                "aic": adf_selected.aic,
                "bic": adf_selected.bic,
                "critical_1pct": adf_selected.critical_1pct,
                "critical_5pct": adf_selected.critical_5pct,
                "critical_10pct": adf_selected.critical_10pct,
                "p_value_range": adf_selected.p_value_range,
                "reject_unit_root_5pct": adf_selected.reject_5pct,
            }
        )
        for candidate in adf_candidates:
            adf_lag_rows.append(
                {
                    "series": key,
                    "lag": candidate.lag,
                    "nobs": candidate.nobs,
                    "adf_statistic": candidate.statistic,
                    "aic": candidate.aic,
                    "bic": candidate.bic,
                }
            )

        kpss = kpss_test(values)
        kpss_rows.append(
            {
                "series": key,
                "regression": "constant",
                "nlags": kpss.nlags,
                "nobs": kpss.nobs,
                "kpss_statistic": kpss.statistic,
                "long_run_variance": kpss.long_run_variance,
                "critical_1pct": kpss.critical_1pct,
                "critical_5pct": kpss.critical_5pct,
                "critical_10pct": kpss.critical_10pct,
                "p_value_range": kpss.p_value_range,
                "reject_stationarity_5pct": kpss.reject_5pct,
            }
        )

        rolling_summary = build_rolling_summary(values, rolling_means, rolling_stds)
        rolling_row = {
            "series": key,
            "window_observations": ROLLING_WINDOW,
            "window_label": "1 day",
            **rolling_summary,
        }
        rolling_rows.append(rolling_row)

    write_rows_csv(
        tables_dir / "phase3_adf_results.csv",
        adf_rows,
        [
            "series",
            "regression",
            "selected_lag",
            "nobs",
            "adf_statistic",
            "gamma",
            "std_error",
            "aic",
            "bic",
            "critical_1pct",
            "critical_5pct",
            "critical_10pct",
            "p_value_range",
            "reject_unit_root_5pct",
        ],
    )
    write_rows_csv(
        tables_dir / "phase3_adf_lag_selection.csv",
        adf_lag_rows,
        ["series", "lag", "nobs", "adf_statistic", "aic", "bic"],
    )
    write_rows_csv(
        tables_dir / "phase3_kpss_results.csv",
        kpss_rows,
        [
            "series",
            "regression",
            "nlags",
            "nobs",
            "kpss_statistic",
            "long_run_variance",
            "critical_1pct",
            "critical_5pct",
            "critical_10pct",
            "p_value_range",
            "reject_stationarity_5pct",
        ],
    )
    write_rows_csv(
        tables_dir / "phase3_rolling_summary.csv",
        rolling_rows,
        [
            "series",
            "window_observations",
            "window_label",
            "raw_mean_first_half",
            "raw_mean_second_half",
            "raw_mean_difference",
            "raw_std_first_half",
            "raw_std_second_half",
            "raw_std_ratio_second_over_first",
            "rolling_mean_min",
            "rolling_mean_max",
            "rolling_std_min",
            "rolling_std_max",
        ],
    )
    print("FASE 3 - ESTACIONARIEDAD Y TRANSFORMACIONES")
    print("=" * 72)
    print(f"Filas analizadas: {len(rows):,}")
    print(f"Series: {', '.join(spec['key'] for spec in SERIES_SPECS)}")
    print(f"Ventana rolling: {ROLLING_WINDOW} observaciones")
    print(f"Tablas: {tables_dir}")
    print(f"Figuras: {figures_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
