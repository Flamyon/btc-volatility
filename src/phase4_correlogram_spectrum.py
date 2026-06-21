"""Fase 4: correlograma y espectro."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
import html
import math

from data_loading import write_rows_csv
from spectral import (
    autocorrelation_fft,
    pacf_levinson,
    periodogram_fft,
    power_at_reference_periods,
    spectral_peaks,
)


SERIES_SPECS = [
    {
        "key": "r",
        "column": "r",
        "label": "Retornos logaritmicos r",
        "color": "#5f6b7a",
    },
    {
        "key": "abs_r",
        "column": "abs_r",
        "label": "Retornos absolutos |r|",
        "color": "#b45f06",
    },
    {
        "key": "log_rv_past_12",
        "column": "log_rv_past_12",
        "label": "v_t = log_rv_past_12",
        "color": "#8a4f9f",
    },
]

ACF_MAIN_LAGS = 288
ACF_EXTENDED_LAGS = 2016
PACF_LAGS = 288
REFERENCE_PERIODS = [12.0, 288.0, 2016.0]


def read_series(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Lee las tres series necesarias para la Fase 4."""
    series = {spec["key"]: [] for spec in SERIES_SPECS}
    times: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(row["open_time"])
            for spec in SERIES_SPECS:
                series[spec["key"]].append(float(row[spec["column"]]))
    return times, series


def correlogram_summary(acf: list[float], pacf: list[float], band: float) -> dict[str, Any]:
    """Resume patrones principales del correlograma."""
    acf_lags = list(range(1, len(acf)))
    significant_acf = [lag for lag in acf_lags if abs(acf[lag]) > band]
    first_below_band = next((lag for lag in acf_lags if abs(acf[lag]) <= band), "")
    first_negative = next((lag for lag in acf_lags if acf[lag] < 0), "")
    max_abs_acf_lag = max(acf_lags, key=lambda lag: abs(acf[lag]))
    max_abs_pacf_lag = max(range(1, len(pacf)), key=lambda lag: abs(pacf[lag]))
    return {
        "acf_lag_1": acf[1],
        "acf_lag_12": acf[12],
        "acf_lag_288": acf[288],
        "acf_lag_2016": acf[2016],
        "acf_significance_band": band,
        "acf_significant_lags_1_288": sum(1 for lag in range(1, 289) if abs(acf[lag]) > band),
        "acf_significant_lags_1_2016": len(significant_acf),
        "acf_first_lag_abs_below_band": first_below_band,
        "acf_first_negative_lag": first_negative,
        "acf_max_abs_lag_1_2016": max_abs_acf_lag,
        "acf_max_abs_value_1_2016": acf[max_abs_acf_lag],
        "pacf_lag_1": pacf[1],
        "pacf_lag_12": pacf[12],
        "pacf_lag_288": pacf[288],
        "pacf_significant_lags_1_288": sum(1 for lag in range(1, 289) if abs(pacf[lag]) > band),
        "pacf_max_abs_lag_1_288": max_abs_pacf_lag,
        "pacf_max_abs_value_1_288": pacf[max_abs_pacf_lag],
    }


def write_correlogram_svg(
    path: Path,
    values: list[float],
    title: str,
    y_label: str,
    significance_band: float,
    color: str,
    reference_lags: list[int] | None = None,
    first_lag: int = 0,
    width: int = 1120,
    height: int = 430,
) -> None:
    """Grafico de barras para ACF/PACF."""
    reference_lags = reference_lags or []
    if not 0 <= first_lag < len(values):
        raise ValueError("first_lag debe estar dentro del rango de valores")
    plotted_lags = list(range(first_lag, len(values)))
    plotted_values = values[first_lag:]
    y_min = min(min(plotted_values), -significance_band) * 1.08
    y_max = max(max(plotted_values), significance_band) * 1.08
    if y_min == y_max:
        y_min, y_max = -1.0, 1.0
    y_min = min(y_min, -0.02)
    y_max = max(y_max, 0.02)

    margin = {"left": 82, "right": 30, "top": 58, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def x_coord(lag: int) -> float:
        return margin["left"] + plot_width * (lag - first_lag) / max(1, len(values) - 1 - first_lag)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    zero_y = y_coord(0.0)
    upper_band_y = y_coord(significance_band)
    lower_band_y = y_coord(-significance_band)
    bar_width = max(0.55, min(3.0, plot_width / max(1, len(plotted_values))))

    elements = [
        _svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width/2:.1f}" y="26" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
            f"{_esc(title)}</text>"
        ),
    ]

    for tick in _ticks(y_min, y_max, 5):
        y = y_coord(tick)
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="#e8e8e8" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{tick:.3g}</text>'
        )

    elements.append(
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222222" stroke-width="1"/>'
    )
    elements.append(
        f'<line x1="{margin["left"]}" y1="{zero_y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
        f'y2="{zero_y:.2f}" stroke="#222222" stroke-width="1"/>'
    )
    for band_y in [upper_band_y, lower_band_y]:
        elements.append(
            f'<line x1="{margin["left"]}" y1="{band_y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{band_y:.2f}" stroke="#a33" stroke-width="1" stroke-dasharray="5,4"/>'
        )

    for lag in reference_lags:
        if first_lag <= lag < len(values):
            x = x_coord(lag)
            elements.append(
                f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" '
                f'y2="{margin["top"] + plot_height:.2f}" stroke="#777777" '
                f'stroke-width="1" stroke-dasharray="3,4"/>'
            )
            elements.append(
                f'<text x="{x:.2f}" y="{margin["top"] - 8}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="10" fill="#555555">{lag}</text>'
            )

    for lag, value in zip(plotted_lags, plotted_values):
        x = x_coord(lag)
        y = y_coord(value)
        y1, y2 = sorted([zero_y, y])
        elements.append(
            f'<rect x="{x - bar_width/2:.2f}" y="{y1:.2f}" width="{bar_width:.2f}" '
            f'height="{max(0.8, y2 - y1):.2f}" fill="{color}" opacity="0.82"/>'
        )

    tick_lags = [lag for lag in _x_tick_lags(len(values) - 1) if lag >= first_lag]
    if first_lag not in tick_lags:
        tick_lags.insert(0, first_lag)
    for lag in tick_lags:
        x = x_coord(lag)
        elements.append(
            f'<text x="{x:.2f}" y="{height - 28}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11">{lag}</text>'
        )
    elements.append(
        f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 8}" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Retardo</text>'
    )
    elements.append(
        f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{_esc(y_label)}</text>'
    )
    elements.append(
        f'<text x="{margin["left"] + 8}" y="{height - 12}" '
        f'font-family="Arial, sans-serif" font-size="11" fill="#8a2222">'
        f"Bandas orientativas ±{significance_band:.4g}</text>"
    )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def write_periodogram_svg(
    path: Path,
    spectrum: list[dict[str, float]],
    title: str,
    color: str,
    width: int = 1120,
    height: int = 460,
    max_points: int = 6000,
) -> None:
    """Grafico de periodograma en funcion del periodo equivalente."""
    filtered = [
        row
        for row in spectrum
        if 2.0 <= row["period_lags"] <= 4032.0 and row["power"] > 0.0
    ]
    filtered.sort(key=lambda row: row["period_lags"])
    points = _downsample_spectrum(filtered, max_points=max_points)
    x_values = [math.log10(row["period_lags"]) for row in points]
    y_values = [math.log10(row["power"]) for row in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = _expanded_range(min(y_values), max(y_values))

    margin = {"left": 82, "right": 34, "top": 58, "bottom": 64}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def x_coord(period_lags: float) -> float:
        return margin["left"] + plot_width * (math.log10(period_lags) - x_min) / (x_max - x_min)

    def y_coord(log_power: float) -> float:
        return margin["top"] + plot_height - plot_height * (log_power - y_min) / (y_max - y_min)

    elements = [
        _svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width/2:.1f}" y="26" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
            f"{_esc(title)}</text>"
        ),
    ]

    for tick in _ticks(y_min, y_max, 5):
        y = y_coord(tick)
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="#e8e8e8" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{tick:.2f}</text>'
        )

    elements.append(
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222222" stroke-width="1"/>'
    )

    polyline = " ".join(
        f"{x_coord(row['period_lags']):.2f},{y_coord(math.log10(row['power'])):.2f}"
        for row in points
    )
    elements.append(
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="1.1" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    for period, label in [(12.0, "1h"), (288.0, "1d"), (2016.0, "1w")]:
        x = x_coord(period)
        elements.append(
            f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" '
            f'y2="{margin["top"] + plot_height:.2f}" stroke="#777777" '
            f'stroke-width="1" stroke-dasharray="4,4"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{margin["top"] - 8}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#555555">{label}</text>'
        )

    for period in [2, 6, 12, 48, 288, 2016, 4032]:
        if 2 <= period <= 4032:
            x = x_coord(period)
            elements.append(
                f'<text x="{x:.2f}" y="{height - 32}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11">{period:g}</text>'
            )
    elements.append(
        f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">'
        "Periodo equivalente en retardos de 5 minutos (escala log)</text>"
    )
    elements.append(
        f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">'
        "log10 potencia</text>"
    )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")




def _group_by_series(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["series"], []).append(row)
    return grouped


def _downsample_spectrum(rows: list[dict[str, float]], max_points: int) -> list[dict[str, float]]:
    if len(rows) <= max_points:
        return rows
    bucket_size = math.ceil(len(rows) / max_points)
    selected: list[dict[str, float]] = []
    for start in range(0, len(rows), bucket_size):
        bucket = rows[start : start + bucket_size]
        if not bucket:
            continue
        selected.append(max(bucket, key=lambda row: row["power"]))
    selected.sort(key=lambda row: row["period_lags"])
    return selected


def _rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}"
        else:
            rounded[key] = value
    return rounded


def _expanded_range(y_min: float, y_max: float) -> tuple[float, float]:
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        return y_min - delta, y_max + delta
    padding = 0.05 * (y_max - y_min)
    return y_min - padding, y_max + padding


def _ticks(y_min: float, y_max: float, count: int) -> list[float]:
    return [y_min + (y_max - y_min) * index / (count - 1) for index in range(count)]


def _x_tick_lags(max_lag: int) -> list[int]:
    if max_lag <= 288:
        return [0, 12, 48, 96, 144, 216, 288]
    return [0, 288, 576, 1008, 1440, 2016]


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 4: correlograma y espectro.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _, series = read_series(args.input)
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    acf_rows: list[dict[str, Any]] = []
    pacf_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in SERIES_SPECS:
        key = spec["key"]
        values = series[key]
        n = len(values)
        band = 1.96 / math.sqrt(n)

        print(f"Calculando ACF/PACF/espectro para {key} ({n:,} observaciones)...")
        acf = autocorrelation_fft(values, ACF_EXTENDED_LAGS)
        pacf = pacf_levinson(acf, PACF_LAGS)
        spectrum = periodogram_fft(values)
        peaks = spectral_peaks(spectrum, top_n=12)
        references = power_at_reference_periods(spectrum, REFERENCE_PERIODS)

        for lag, value in enumerate(acf):
            acf_rows.append({"series": key, "lag": lag, "acf": value})
        for lag, value in enumerate(pacf):
            pacf_rows.append({"series": key, "lag": lag, "pacf": value})
        for peak in peaks:
            peak_rows.append({"series": key, **peak})
        for row in references:
            reference_rows.append({"series": key, **row})

        summary = {"series": key, "n": n, **correlogram_summary(acf, pacf, band)}
        summary_rows.append(summary)

        write_correlogram_svg(
            figures_dir / f"phase4_{key}_acf_288.svg",
            acf[: ACF_MAIN_LAGS + 1],
            title=f"ACF hasta 288 retardos - {spec['label']}",
            y_label="ACF",
            significance_band=band,
            color=spec["color"],
            reference_lags=[12, 288],
        )
        write_correlogram_svg(
            figures_dir / f"phase4_{key}_acf_2016.svg",
            acf,
            title=f"ACF hasta 2016 retardos - {spec['label']}",
            y_label="ACF",
            significance_band=band,
            color=spec["color"],
            reference_lags=[12, 288, 2016],
        )
        write_correlogram_svg(
            figures_dir / f"phase4_{key}_pacf_288.svg",
            pacf,
            title=f"PACF hasta 288 retardos - {spec['label']}",
            y_label="PACF",
            significance_band=band,
            color=spec["color"],
            reference_lags=[12, 288],
        )
        write_periodogram_svg(
            figures_dir / f"phase4_{key}_periodogram.svg",
            spectrum,
            title=f"Periodograma - {spec['label']}",
            color=spec["color"],
        )

    write_rows_csv(tables_dir / "phase4_acf_values.csv", acf_rows, ["series", "lag", "acf"])
    write_rows_csv(tables_dir / "phase4_pacf_values.csv", pacf_rows, ["series", "lag", "pacf"])
    write_rows_csv(
        tables_dir / "phase4_spectral_peaks.csv",
        peak_rows,
        [
            "series",
            "rank",
            "period_lags",
            "period_hours",
            "period_days",
            "frequency_cycles_per_observation",
            "power",
            "nearest_reference",
        ],
    )
    write_rows_csv(
        tables_dir / "phase4_spectral_reference_power.csv",
        reference_rows,
        [
            "series",
            "reference_label",
            "reference_period_lags",
            "nearest_period_lags",
            "nearest_period_hours",
            "nearest_period_days",
            "power",
        ],
    )
    write_rows_csv(
        tables_dir / "phase4_correlogram_summary.csv",
        summary_rows,
        [
            "series",
            "n",
            "acf_lag_1",
            "acf_lag_12",
            "acf_lag_288",
            "acf_lag_2016",
            "acf_significance_band",
            "acf_significant_lags_1_288",
            "acf_significant_lags_1_2016",
            "acf_first_lag_abs_below_band",
            "acf_first_negative_lag",
            "acf_max_abs_lag_1_2016",
            "acf_max_abs_value_1_2016",
            "pacf_lag_1",
            "pacf_lag_12",
            "pacf_lag_288",
            "pacf_significant_lags_1_288",
            "pacf_max_abs_lag_1_288",
            "pacf_max_abs_value_1_288",
        ],
    )
    print("FASE 4 - CORRELOGRAMA Y ESPECTRO")
    print("=" * 72)
    print(f"Series analizadas: {', '.join(spec['key'] for spec in SERIES_SPECS)}")
    print(f"Tablas: {tables_dir}")
    print(f"Figuras: {figures_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
