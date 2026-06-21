"""Fase 7: contrastes de dependencia no lineal residual."""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from linear_filtering import ljung_box
from nonlinear_tests import (
    arch_lm_yule_walker,
    bds_test_window,
    shuffle_squared_acf_comparison,
    zscore,
)
from phase4_correlogram_spectrum import write_correlogram_svg
from spectral import autocorrelation_fft


SERIES_SPECS = [
    {
        "key": "z_log_rv_past_12",
        "column": "z_log_rv_past_12",
        "label": "z_t = z_log_rv_past_12",
        "color": "#8a4f9f",
    },
    {
        "key": "ar_residual",
        "column": "residual",
        "label": "e_t = residuos AR(49)",
        "color": "#60758a",
    },
]

LJUNG_BOX_LAGS = [12, 24, 48, 96, 288, 2016]
ARCH_LM_LAGS = [12, 24, 48, 96, 288]
BDS_EMBEDDING_DIMS = [2, 3, 4]
BDS_EPSILON_MULTIPLIERS = [0.5, 1.0, 1.5]
SHUFFLE_LAGS = [1, 12, 288]
WINDOW_SIZE = 3000
N_SHUFFLES = 50
RANDOM_SEED = 20260601
ACF_FIGURE_LAGS = 288
RESIDUAL_OFFSET_FROM_FEATURES = 49


def read_residual_series(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Lee z_t y residuos AR ya alineados de Fase 6."""
    times: list[str] = []
    series = {spec["key"]: [] for spec in SERIES_SPECS}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"open_time", "z_log_rv_past_12", "residual"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
        for row in reader:
            times.append(row["open_time"])
            for spec in SERIES_SPECS:
                series[spec["key"]].append(float(row[spec["column"]]))
    if not times:
        raise ValueError(f"No hay datos en {path}")
    return times, series


def read_phase5_windows(path: Path) -> list[dict[str, Any]]:
    """Lee ventanas de Fase 5 si existen."""
    if not path.exists():
        return []
    windows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            parsed["start_index"] = int(row["start_index"])
            parsed["end_index_exclusive"] = int(row["end_index_exclusive"])
            parsed["n"] = int(row["n"])
            windows.append(parsed)
    return windows


def select_windows(
    times: list[str],
    z_values: list[float],
    phase5_windows: list[dict[str, Any]],
    window_size: int,
) -> list[dict[str, Any]]:
    """Selecciona ventanas BDS de forma reproducible."""
    n = len(z_values)
    if n < window_size:
        raise ValueError("La serie residual es menor que la ventana solicitada")

    if phase5_windows:
        return expand_phase5_windows(times, z_values, phase5_windows, window_size)
    return select_windows_from_scratch(times, z_values, window_size)


def expand_phase5_windows(
    times: list[str],
    z_values: list[float],
    phase5_windows: list[dict[str, Any]],
    window_size: int,
) -> list[dict[str, Any]]:
    """Reutiliza los centros de Fase 5 y amplia cada ventana a window_size."""
    time_to_index = {time: index for index, time in enumerate(times)}
    windows: list[dict[str, Any]] = []

    for previous in phase5_windows:
        name = previous["window"]
        if name == "recent":
            start = len(times) - window_size
            center_index = start + window_size // 2
            source = "ultimas observaciones disponibles, ampliando la ventana reciente de Fase 5"
        elif name == "high_volatility" and previous.get("reference_time"):
            center_index = time_to_index.get(previous["reference_time"])
            if center_index is None:
                center_index = index_from_phase5_midpoint(previous)
            start = clamp(center_index - window_size // 2, 0, len(times) - window_size)
            source = "centrada en el episodio de alta volatilidad identificado en Fase 5"
        else:
            start_time = previous["start_time"]
            end_time = previous["end_time"]
            if start_time in time_to_index and end_time in time_to_index:
                center_index = (time_to_index[start_time] + time_to_index[end_time]) // 2
            else:
                center_index = index_from_phase5_midpoint(previous)
            start = clamp(center_index - window_size // 2, 0, len(times) - window_size)
            source = "centro reutilizado de la ventana seleccionada en Fase 5"

        end = start + window_size
        values = z_values[start:end]
        windows.append(
            {
                "window": name,
                "description": previous.get("description", name),
                "selection_method": source,
                "start_index_residual_series": start,
                "end_index_exclusive_residual_series": end,
                "start_index_feature_series_approx": start + RESIDUAL_OFFSET_FROM_FEATURES,
                "end_index_exclusive_feature_series_approx": end + RESIDUAL_OFFSET_FROM_FEATURES,
                "start_time": times[start],
                "end_time": times[end - 1],
                "center_time": times[center_index],
                "n": window_size,
                "mean_z_log_rv_past_12": sum(values) / len(values),
                "min_z_log_rv_past_12": min(values),
                "max_z_log_rv_past_12": max(values),
            }
        )

    order = {"middle": 0, "recent": 1, "high_volatility": 2, "quiet": 3}
    windows.sort(key=lambda row: order.get(row["window"], 99))
    return windows


def index_from_phase5_midpoint(window: dict[str, Any]) -> int:
    midpoint_feature_index = (window["start_index"] + window["end_index_exclusive"] - 1) // 2
    return max(0, midpoint_feature_index - RESIDUAL_OFFSET_FROM_FEATURES)


def select_windows_from_scratch(
    times: list[str],
    z_values: list[float],
    window_size: int,
) -> list[dict[str, Any]]:
    """Selecciona ventanas si no existe la tabla de Fase 5."""
    n = len(z_values)
    rolling_means = rolling_window_means(z_values, window_size)
    quiet_target = percentile(sorted(rolling_means), 0.10)
    quiet_start = min(
        range(len(rolling_means)),
        key=lambda index: abs(rolling_means[index] - quiet_target),
    )
    high_center = max(range(n), key=lambda index: z_values[index])
    high_start = clamp(high_center - window_size // 2, 0, n - window_size)
    middle_start = n // 2 - window_size // 2
    recent_start = n - window_size
    raw_windows = [
        ("middle", "Ventana continua representativa", middle_start, "bloque central de la serie"),
        ("recent", "Ventana reciente", recent_start, "ultimas observaciones disponibles"),
        (
            "high_volatility",
            "Ventana de alta volatilidad",
            high_start,
            "ventana centrada en el maximo de z_log_rv_past_12",
        ),
        (
            "quiet",
            "Ventana tranquila",
            quiet_start,
            "media rolling mas cercana al percentil 10",
        ),
    ]
    windows: list[dict[str, Any]] = []
    for name, description, start, method in raw_windows:
        end = start + window_size
        values = z_values[start:end]
        windows.append(
            {
                "window": name,
                "description": description,
                "selection_method": method,
                "start_index_residual_series": start,
                "end_index_exclusive_residual_series": end,
                "start_index_feature_series_approx": start + RESIDUAL_OFFSET_FROM_FEATURES,
                "end_index_exclusive_feature_series_approx": end + RESIDUAL_OFFSET_FROM_FEATURES,
                "start_time": times[start],
                "end_time": times[end - 1],
                "center_time": times[start + window_size // 2],
                "n": window_size,
                "mean_z_log_rv_past_12": sum(values) / len(values),
                "min_z_log_rv_past_12": min(values),
                "max_z_log_rv_past_12": max(values),
            }
        )
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


def ljung_box_squared_rows(series: dict[str, list[float]]) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Ljung-Box sobre cuadrados de z_t y e_t."""
    rows: list[dict[str, Any]] = []
    square_acfs: dict[str, list[float]] = {}
    max_lag = max(LJUNG_BOX_LAGS)
    for spec in SERIES_SPECS:
        key = spec["key"]
        values = series[key]
        squares = [value * value for value in values]
        acf = autocorrelation_fft(squares, max_lag)
        square_acfs[key] = acf
        for row in ljung_box(acf, len(squares), LJUNG_BOX_LAGS, fitted_order=0):
            p_value = row["p_value"]
            rows.append(
                {
                    "series": key,
                    "lag": row["lag"],
                    "q_stat": row["q_stat"],
                    "p_value": p_value,
                    "reject_5pct": bool(p_value < 0.05) if isinstance(p_value, float) else "",
                    "decision_5pct": "rechaza H0" if isinstance(p_value, float) and p_value < 0.05 else "no rechaza H0",
                }
            )
    return rows, square_acfs


def arch_lm_rows(residual_values: list[float]) -> list[dict[str, Any]]:
    """ARCH LM aproximado sobre residuos AR."""
    rows: list[dict[str, Any]] = []
    for row in arch_lm_yule_walker(residual_values, ARCH_LM_LAGS):
        p_value = float(row["p_value"])
        rows.append(
            {
                "series": "ar_residual",
                "lag": row["lag"],
                "nobs_effective": row["nobs_effective"],
                "lm_stat": row["lm_stat"],
                "r_squared_yw": row["r_squared_yw"],
                "p_value": p_value,
                "reject_5pct": p_value < 0.05,
                "decision_5pct": "rechaza H0" if p_value < 0.05 else "no rechaza H0",
                "implementation_note": "aproximacion Yule-Walker para evitar OLS masivo",
            }
        )
    return rows


def bds_rows(
    series: dict[str, list[float]],
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """BDS sobre ventanas normalizadas."""
    rows: list[dict[str, Any]] = []
    for spec in SERIES_SPECS:
        key = spec["key"]
        values = series[key]
        for window in windows:
            start = int(window["start_index_residual_series"])
            end = int(window["end_index_exclusive_residual_series"])
            normalized = zscore(values[start:end])
            print(f"BDS: {key}, ventana {window['window']}, n={len(normalized):,}")
            results = bds_test_window(
                normalized,
                embedding_dims=BDS_EMBEDDING_DIMS,
                epsilons=BDS_EPSILON_MULTIPLIERS,
            )
            for result in results:
                p_value = result.p_value
                rows.append(
                    {
                        "series": key,
                        "window": window["window"],
                        "start_time": window["start_time"],
                        "end_time": window["end_time"],
                        "n_window": len(normalized),
                        "embedding_dim_m": result.embedding_dim,
                        "epsilon_multiplier": result.epsilon,
                        "epsilon": result.epsilon,
                        "bds_statistic": result.statistic,
                        "p_value": p_value,
                        "reject_5pct": p_value < 0.05 if math.isfinite(p_value) else "",
                        "decision_5pct": (
                            "rechaza H0"
                            if math.isfinite(p_value) and p_value < 0.05
                            else "no rechaza H0"
                        ),
                        "correlation_sum_1": result.correlation_sum_1,
                        "correlation_sum_m": result.correlation_sum_m,
                        "variance": result.variance,
                    }
                )
    return rows


def shuffle_rows(
    series: dict[str, list[float]],
    windows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Comparacion de autocorrelacion de cuadrados original vs barajado."""
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for spec in SERIES_SPECS:
        key = spec["key"]
        values = series[key]
        for window in windows:
            start = int(window["start_index_residual_series"])
            end = int(window["end_index_exclusive_residual_series"])
            normalized = zscore(values[start:end])
            comparison = shuffle_squared_acf_comparison(
                normalized,
                lags=SHUFFLE_LAGS,
                n_shuffles=N_SHUFFLES,
                rng=rng,
            )
            for row in comparison:
                percentile_value = float(row["empirical_percentile_original"])
                rows.append(
                    {
                        "series": key,
                        "window": window["window"],
                        "start_time": window["start_time"],
                        "end_time": window["end_time"],
                        "statistic": row["statistic"],
                        "lag": row["lag"],
                        "original_value": row["original_value"],
                        "shuffle_mean": row["shuffle_mean"],
                        "shuffle_std": row["shuffle_std"],
                        "empirical_percentile_original": percentile_value,
                        "n_shuffles": row["n_shuffles"],
                        "extreme_5pct_two_sided": percentile_value <= 0.05 or percentile_value >= 0.95,
                    }
                )
    return rows


def square_acf_value_rows(square_acfs: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, acf in square_acfs.items():
        for lag, value in enumerate(acf):
            rows.append({"series": key, "lag": lag, "acf_squared": value})
    return rows


def write_bds_pvalue_svg(path: Path, rows: list[dict[str, Any]], width: int = 1180, height: int = 500) -> None:
    """Grafico resumen de p-values BDS en escala -log10."""
    if not rows:
        return
    margin = {"left": 72, "right": 34, "top": 58, "bottom": 116}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    capped_values = [min(16.0, -math.log10(max(float(row["p_value"]), 1e-16))) for row in rows]
    y_max = max(2.0, max(capped_values) * 1.08)

    def x_coord(index: int) -> float:
        return margin["left"] + plot_width * (index + 0.5) / len(rows)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * value / y_max

    colors = {"z_log_rv_past_12": "#8a4f9f", "ar_residual": "#60758a"}
    elements = [
        _svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
            "BDS en ventanas: resumen de p-values</text>"
        ),
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222222" stroke-width="1"/>',
    ]
    for tick in [0.0, 1.30103, 4.0, 8.0, 12.0, 16.0]:
        if tick > y_max:
            continue
        y = y_coord(tick)
        label = "p=0.05" if abs(tick - 1.30103) < 1e-5 else f"{tick:.0f}"
        color = "#a33" if label == "p=0.05" else "#e8e8e8"
        dash = ' stroke-dasharray="5,4"' if label == "p=0.05" else ""
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="{color}" stroke-width="1"{dash}/>'
        )
        elements.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{label}</text>'
        )

    last_group = ""
    for index, row in enumerate(rows):
        value = capped_values[index]
        x = x_coord(index)
        y = y_coord(value)
        color = colors.get(str(row["series"]), "#333333")
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}" opacity="0.82"/>'
        )
        group = f"{row['series']}\n{row['window']}"
        if group != last_group:
            elements.append(
                f'<text x="{x:.2f}" y="{height - 62}" text-anchor="end" '
                f'transform="rotate(-45 {x:.2f},{height - 62})" '
                f'font-family="Arial, sans-serif" font-size="10">{_esc(str(row["window"]))}</text>'
            )
            last_group = group

    elements.extend(
        [
            (
                f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 12}" '
                f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">'
                "Puntos por serie, ventana, m y epsilon</text>"
            ),
            (
                f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
                f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">'
                "-log10(p-value), capado en 16</text>"
            ),
            legend_item(920, 28, "#8a4f9f", "z_t"),
            legend_item(1000, 28, "#60758a", "e_t"),
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def write_shuffle_svg(path: Path, rows: list[dict[str, Any]], width: int = 1220, height: int = 520) -> None:
    """Grafico original vs media de barajadas para ACF de cuadrados."""
    if not rows:
        return
    margin = {"left": 76, "right": 34, "top": 60, "bottom": 128}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    values = []
    for row in rows:
        original = float(row["original_value"])
        mean = float(row["shuffle_mean"])
        std = float(row["shuffle_std"])
        values.extend([original, mean - 2.0 * std, mean + 2.0 * std])
    y_min, y_max = expanded_range(min(values), max(values))

    def x_coord(index: int) -> float:
        return margin["left"] + plot_width * (index + 0.5) / len(rows)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = [
        _svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
            "Autocorrelacion de cuadrados: original vs barajadas</text>"
        ),
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222222" stroke-width="1"/>',
    ]
    for tick in ticks(y_min, y_max, 6):
        y = y_coord(tick)
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="#e8e8e8" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{tick:.3g}</text>'
        )

    for index, row in enumerate(rows):
        x = x_coord(index)
        original = float(row["original_value"])
        mean = float(row["shuffle_mean"])
        std = float(row["shuffle_std"])
        y_mean = y_coord(mean)
        y_low = y_coord(mean - 2.0 * std)
        y_high = y_coord(mean + 2.0 * std)
        elements.append(
            f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}" '
            f'stroke="#777777" stroke-width="1.1"/>'
        )
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y_mean:.2f}" r="3" fill="#777777" opacity="0.75"/>'
        )
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y_coord(original):.2f}" r="4" fill="#b45f06" opacity="0.9"/>'
        )
        label = f"{short_series(str(row['series']))}/{row['window']}/L{row['lag']}"
        if index % 2 == 0:
            elements.append(
                f'<text x="{x:.2f}" y="{height - 68}" text-anchor="end" '
                f'transform="rotate(-55 {x:.2f},{height - 68})" '
                f'font-family="Arial, sans-serif" font-size="9.5">{_esc(label)}</text>'
            )

    elements.extend(
        [
            legend_item(835, 30, "#b45f06", "original"),
            legend_item(940, 30, "#777777", "media barajada +/- 2 sd"),
            (
                f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" '
                f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">'
                "Grupos: serie / ventana / retardo</text>"
            ),
            (
                f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
                f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">'
                "ACF de cuadrados</text>"
            ),
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")




def summarize_bds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_by_pair(rows, "series", "window")
    summary: list[dict[str, Any]] = []
    for (series, window), group in grouped.items():
        p_values = [float(row["p_value"]) for row in group if math.isfinite(float(row["p_value"]))]
        summary.append(
            {
                "series": series,
                "window": window,
                "tests": len(group),
                "rejections_5pct": sum(1 for row in group if row["reject_5pct"] is True),
                "min_p_value": min(p_values) if p_values else float("nan"),
                "max_p_value": max(p_values) if p_values else float("nan"),
            }
        )
    summary.sort(key=lambda row: (row["series"], row["window"]))
    return summary


def summarize_shuffle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_by_pair(rows, "series", "window")
    summary: list[dict[str, Any]] = []
    for (series, window), group in grouped.items():
        summary.append(
            {
                "series": series,
                "window": window,
                "statistics": len(group),
                "extreme_5pct_two_sided": sum(1 for row in group if row["extreme_5pct_two_sided"]),
                "mean_abs_original": sum(abs(float(row["original_value"])) for row in group) / len(group),
                "mean_abs_shuffle_mean": sum(abs(float(row["shuffle_mean"])) for row in group) / len(group),
            }
        )
    summary.sort(key=lambda row: (row["series"], row["window"]))
    return summary


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


def group_by_pair(
    rows: list[dict[str, Any]],
    first_key: str,
    second_key: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row[first_key]), str(row[second_key])), []).append(row)
    return grouped


def rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}"
        else:
            rounded[key] = value
    return rounded


def expanded_range(y_min: float, y_max: float) -> tuple[float, float]:
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        return y_min - delta, y_max + delta
    padding = 0.08 * (y_max - y_min)
    return y_min - padding, y_max + padding


def ticks(y_min: float, y_max: float, count: int) -> list[float]:
    return [y_min + (y_max - y_min) * index / (count - 1) for index in range(count)]


def short_series(series: str) -> str:
    return "z" if series == "z_log_rv_past_12" else "e"


def legend_item(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<g><circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
        f'<text x="{x + 10:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
        f'font-size="12">{_esc(label)}</text></g>'
    )


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 7: contrastes no lineales.")
    parser.add_argument(
        "--residuals",
        type=Path,
        default=Path("reports/tables/phase6_residual_series.csv"),
        help="CSV con z_t y residuos AR de Fase 6.",
    )
    parser.add_argument(
        "--phase5-windows",
        type=Path,
        default=Path("reports/tables/phase5_selected_windows.csv"),
        help="CSV de ventanas seleccionadas en Fase 5.",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    times, series = read_residual_series(args.residuals)
    print(f"Leidas {len(times):,} observaciones alineadas desde {args.residuals}")
    phase5_windows = read_phase5_windows(args.phase5_windows)
    windows = select_windows(times, series["z_log_rv_past_12"], phase5_windows, WINDOW_SIZE)

    ljung_rows, square_acfs = ljung_box_squared_rows(series)
    arch_rows = arch_lm_rows(series["ar_residual"])

    band = 1.96 / math.sqrt(len(times))
    write_correlogram_svg(
        figures_dir / "phase7_acf_z_squared.svg",
        square_acfs["z_log_rv_past_12"][: ACF_FIGURE_LAGS + 1],
        title="ACF de z_t^2 hasta 288 retardos",
        y_label="ACF",
        significance_band=band,
        color="#8a4f9f",
        reference_lags=[12, 288],
        first_lag=1,
    )
    write_correlogram_svg(
        figures_dir / "phase7_acf_residual_squared.svg",
        square_acfs["ar_residual"][: ACF_FIGURE_LAGS + 1],
        title="ACF de residuos AR(49) al cuadrado hasta 288 retardos",
        y_label="ACF",
        significance_band=band,
        color="#60758a",
        reference_lags=[12, 288],
        first_lag=1,
    )

    bds_result_rows = bds_rows(series, windows)
    shuffled_rows = shuffle_rows(series, windows, RANDOM_SEED)

    write_bds_pvalue_svg(figures_dir / "phase7_bds_pvalues.svg", bds_result_rows)
    write_shuffle_svg(figures_dir / "phase7_shuffle_squared_acf.svg", shuffled_rows)

    write_rows_csv(
        tables_dir / "phase7_windows.csv",
        windows,
        [
            "window",
            "description",
            "selection_method",
            "start_index_residual_series",
            "end_index_exclusive_residual_series",
            "start_index_feature_series_approx",
            "end_index_exclusive_feature_series_approx",
            "start_time",
            "end_time",
            "center_time",
            "n",
            "mean_z_log_rv_past_12",
            "min_z_log_rv_past_12",
            "max_z_log_rv_past_12",
        ],
    )
    write_rows_csv(
        tables_dir / "phase7_ljungbox_squared.csv",
        ljung_rows,
        ["series", "lag", "q_stat", "p_value", "reject_5pct", "decision_5pct"],
    )
    write_rows_csv(
        tables_dir / "phase7_arch_lm.csv",
        arch_rows,
        [
            "series",
            "lag",
            "nobs_effective",
            "lm_stat",
            "r_squared_yw",
            "p_value",
            "reject_5pct",
            "decision_5pct",
            "implementation_note",
        ],
    )
    write_rows_csv(
        tables_dir / "phase7_bds_results.csv",
        bds_result_rows,
        [
            "series",
            "window",
            "start_time",
            "end_time",
            "n_window",
            "embedding_dim_m",
            "epsilon_multiplier",
            "epsilon",
            "bds_statistic",
            "p_value",
            "reject_5pct",
            "decision_5pct",
            "correlation_sum_1",
            "correlation_sum_m",
            "variance",
        ],
    )
    write_rows_csv(
        tables_dir / "phase7_shuffle_results.csv",
        shuffled_rows,
        [
            "series",
            "window",
            "start_time",
            "end_time",
            "statistic",
            "lag",
            "original_value",
            "shuffle_mean",
            "shuffle_std",
            "empirical_percentile_original",
            "n_shuffles",
            "extreme_5pct_two_sided",
        ],
    )
    write_rows_csv(
        tables_dir / "phase7_squared_acf_values.csv",
        square_acf_value_rows(square_acfs),
        ["series", "lag", "acf_squared"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
