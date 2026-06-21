"""Fase 6: filtrado lineal de v_t = log_rv_past_12."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
import html
import math

from data_loading import write_rows_csv
from linear_filtering import (
    ARModel,
    ar_residuals,
    ljung_box,
    select_ar_yule_walker,
    standardize_train_apply,
)
from phase4_correlogram_spectrum import write_correlogram_svg
from plotting import write_line_svg
from recurrence import recurrence_epsilon_for_rr, write_recurrence_png, zscore
from spectral import autocorrelation_fft, pacf_levinson


SERIES_COLUMN = "log_rv_past_12"
TRAIN_END = "2025-06-30 23:55:00"
MAX_AR_ORDER = 100
ACF_EXTENDED_LAGS = 2016
PACF_LAGS = 288
RECURRENCE_TARGET_RR = 0.05
LJUNG_BOX_LAGS = [12, 48, 288, 2016]


def read_main_series(path: Path) -> tuple[list[str], list[float]]:
    """Lee open_time y v_t."""
    times: list[str] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.append(row["open_time"])
            values.append(float(row[SERIES_COLUMN]))
    return times, values


def train_end_index(times: list[str], train_end: str) -> int:
    """Indice exclusivo del tramo de entrenamiento."""
    index = 0
    while index < len(times) and times[index] <= train_end:
        index += 1
    if index == 0:
        raise ValueError("No hay observaciones de entrenamiento")
    return index


def read_phase5_windows(path: Path) -> list[dict[str, Any]]:
    """Lee las ventanas seleccionadas en Fase 5."""
    windows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["start_index"] = int(row["start_index"])
            parsed["end_index_exclusive"] = int(row["end_index_exclusive"])
            parsed["n"] = int(row["n"])
            windows.append(parsed)
    return windows


def model_rows(models: list[ARModel]) -> list[dict[str, Any]]:
    """Convierte modelos AR a filas CSV."""
    return [
        {
            "p": model.order,
            "nobs_train": model.nobs,
            "innovation_variance": model.innovation_variance,
            "aic": model.aic,
            "bic": model.bic,
        }
        for model in models
    ]


def coefficient_rows(model: ARModel) -> list[dict[str, Any]]:
    """Tabla de coeficientes del AR seleccionado."""
    return [
        {
            "lag": lag,
            "coefficient": coeff,
            "abs_coefficient": abs(coeff),
            "rank_abs": rank,
        }
        for rank, (lag, coeff) in enumerate(
            sorted(
                enumerate(model.coefficients, start=1),
                key=lambda item: abs(item[1]),
                reverse=True,
            ),
            start=1,
        )
    ]


def residual_rows(
    times: list[str],
    values: list[float],
    standardized: list[float],
    fitted: list[float | None],
    residuals: list[float | None],
) -> list[dict[str, Any]]:
    """Serie temporal de residuos."""
    rows: list[dict[str, Any]] = []
    for index, residual in enumerate(residuals):
        if residual is None or fitted[index] is None:
            continue
        rows.append(
            {
                "open_time": times[index],
                "log_rv_past_12": values[index],
                "z_log_rv_past_12": standardized[index],
                "fitted_z": fitted[index],
                "residual": residual,
            }
        )
    return rows


def residual_statistics(residual_values: list[float]) -> dict[str, Any]:
    """Resumen de distribucion de residuos."""
    sorted_values = sorted(residual_values)
    mean = sum(residual_values) / len(residual_values)
    std = sample_std(residual_values)
    centered = [value - mean for value in residual_values]
    m2 = sum(value * value for value in centered) / len(centered)
    m3 = sum(value**3 for value in centered) / len(centered)
    m4 = sum(value**4 for value in centered) / len(centered)
    return {
        "n": len(residual_values),
        "mean": mean,
        "std": std,
        "min": sorted_values[0],
        "p01": percentile(sorted_values, 0.01),
        "p05": percentile(sorted_values, 0.05),
        "p50": percentile(sorted_values, 0.50),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "skewness": m3 / (m2**1.5) if m2 > 0 else 0.0,
        "kurtosis_excess": m4 / (m2 * m2) - 3.0 if m2 > 0 else 0.0,
    }


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


def sample_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def correlogram_summary(
    label: str,
    acf: list[float],
    pacf: list[float],
    nobs: int,
) -> dict[str, Any]:
    band = 1.96 / math.sqrt(nobs)
    return {
        "series": label,
        "nobs": nobs,
        "significance_band": band,
        "acf_lag_1": acf[1],
        "acf_lag_12": acf[12],
        "acf_lag_288": acf[288],
        "acf_lag_2016": acf[2016],
        "acf_significant_lags_1_288": sum(1 for lag in range(1, 289) if abs(acf[lag]) > band),
        "acf_significant_lags_1_2016": sum(1 for lag in range(1, 2017) if abs(acf[lag]) > band),
        "pacf_lag_1": pacf[1],
        "pacf_lag_12": pacf[12],
        "pacf_lag_288": pacf[288],
        "pacf_significant_lags_1_288": sum(1 for lag in range(1, 289) if abs(pacf[lag]) > band),
    }


def write_histogram_svg(
    path: Path,
    values: list[float],
    title: str,
    bins: int = 90,
    width: int = 1000,
    height: int = 420,
) -> dict[str, Any]:
    """Histograma SVG con rango central p0.5-p99.5 y conteo de colas."""
    sorted_values = sorted(values)
    lower = percentile(sorted_values, 0.005)
    upper = percentile(sorted_values, 0.995)
    if lower == upper:
        lower, upper = min(values), max(values)
    counts = [0 for _ in range(bins)]
    below = 0
    above = 0
    for value in values:
        if value < lower:
            below += 1
        elif value > upper:
            above += 1
        else:
            index = min(bins - 1, int((value - lower) / (upper - lower) * bins))
            counts[index] += 1

    max_count = max(counts) if counts else 1
    margin = {"left": 72, "right": 30, "top": 58, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    bar_width = plot_width / bins

    elements = [
        _svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width/2:.1f}" y="26" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
            f"{_esc(title)}</text>"
        ),
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222222" stroke-width="1"/>',
    ]
    for tick in [0, max_count / 2, max_count]:
        y = margin["top"] + plot_height - plot_height * tick / max_count
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="#e8e8e8" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{tick:.0f}</text>'
        )
    for index, count in enumerate(counts):
        x = margin["left"] + index * bar_width
        h = plot_height * count / max_count if max_count > 0 else 0.0
        y = margin["top"] + plot_height - h
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0.8, bar_width - 1):.2f}" '
            f'height="{h:.2f}" fill="#60758a" opacity="0.85"/>'
        )
    for value, label in [(lower, "p0.5"), (0.0, "0"), (upper, "p99.5")]:
        if lower <= value <= upper:
            x = margin["left"] + plot_width * (value - lower) / (upper - lower)
            elements.append(
                f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" '
                f'y2="{margin["top"] + plot_height:.2f}" stroke="#8a2222" '
                f'stroke-width="1" stroke-dasharray="4,4"/>'
            )
            elements.append(
                f'<text x="{x:.2f}" y="{height - 34}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11">{label}</text>'
            )
    elements.append(
        f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">'
        "Residuo estandarizado filtrado</text>"
    )
    elements.append(
        f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Frecuencia</text>'
    )
    elements.append(
        f'<text x="{margin["left"] + 8}" y="{height - 38}" '
        f'font-family="Arial, sans-serif" font-size="11" fill="#555555">'
        f"Rango visual p0.5-p99.5; cola izq={below}, cola der={above}</text>"
    )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")
    return {"histogram_lower": lower, "histogram_upper": upper, "tail_below": below, "tail_above": above}




def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def model_equation(model: ARModel, max_terms: int = 12) -> str:
    terms = [f"{model.intercept:.6g}"]
    for lag, coeff in enumerate(model.coefficients[:max_terms], start=1):
        sign = "+" if coeff >= 0 else "-"
        terms.append(f" {sign} {abs(coeff):.6g} z_(t-{lag})")
    if len(model.coefficients) > max_terms:
        terms.append(f" + ... + phi_{model.order} z_(t-{model.order})")
    return "z_t = " + "".join(terms) + " + e_t"




def _rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}"
        else:
            rounded[key] = value
    return rounded


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 6: filtrado lineal AR.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    times, values = read_main_series(args.input)
    train_end = train_end_index(times, TRAIN_END)
    z_values, train_mean, train_std = standardize_train_apply(values, train_end)
    z_train = z_values[:train_end]

    selected_model, models = select_ar_yule_walker(z_train, MAX_AR_ORDER)
    fitted, residuals = ar_residuals(z_values, selected_model)
    residual_table = residual_rows(times, values, z_values, fitted, residuals)
    residual_values = [float(row["residual"]) for row in residual_table]
    residual_times = [row["open_time"] for row in residual_table]

    reports_dir = args.reports_dir
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"AR seleccionado por BIC: p={selected_model.order}")

    ar_rows = model_rows(models)
    top_models = sorted(ar_rows, key=lambda row: row["bic"])[:10]
    coeff_rows = coefficient_rows(selected_model)
    top_coefficients = coeff_rows[:20]
    residual_stats = residual_statistics(residual_values)
    hist_info = write_histogram_svg(
        figures_dir / "phase6_residuals_histogram.svg",
        residual_values,
        "Distribucion de residuos AR",
    )
    residual_stats.update(hist_info)

    write_line_svg(
        figures_dir / "phase6_residuals_time.svg",
        residual_times,
        residual_values,
        title=f"Residuos AR({selected_model.order}) sobre v_t estandarizada",
        y_label="residuo",
        color="#60758a",
    )

    residual_acf = autocorrelation_fft(residual_values, ACF_EXTENDED_LAGS)
    residual_pacf = pacf_levinson(residual_acf, PACF_LAGS)
    z_acf = autocorrelation_fft(z_values, ACF_EXTENDED_LAGS)
    z_pacf = pacf_levinson(z_acf, PACF_LAGS)
    band = 1.96 / math.sqrt(len(residual_values))

    write_correlogram_svg(
        figures_dir / "phase6_residuals_acf_288.svg",
        residual_acf[:289],
        title=f"ACF residuos AR({selected_model.order}) hasta 288 retardos",
        y_label="ACF",
        significance_band=band,
        color="#60758a",
        reference_lags=[12, 288],
        first_lag=1,
    )
    write_correlogram_svg(
        figures_dir / "phase6_residuals_acf_2016.svg",
        residual_acf,
        title=f"ACF residuos AR({selected_model.order}) hasta 2016 retardos",
        y_label="ACF",
        significance_band=band,
        color="#60758a",
        reference_lags=[12, 288, 2016],
        first_lag=1,
    )
    write_correlogram_svg(
        figures_dir / "phase6_residuals_pacf_288.svg",
        residual_pacf,
        title=f"PACF residuos AR({selected_model.order}) hasta 288 retardos",
        y_label="PACF",
        significance_band=band,
        color="#60758a",
        reference_lags=[12, 288],
        first_lag=1,
    )

    ljung_rows = ljung_box(residual_acf, len(residual_values), LJUNG_BOX_LAGS, selected_model.order)
    correlogram_rows = [
        correlogram_summary("z_log_rv_past_12", z_acf, z_pacf, len(z_values)),
        correlogram_summary("ar_residual", residual_acf, residual_pacf, len(residual_values)),
    ]

    windows_path = tables_dir / "phase5_selected_windows.csv"
    windows = read_phase5_windows(windows_path)
    recurrence_rows: list[dict[str, Any]] = []
    residual_full = [None if value is None else float(value) for value in residuals]
    for window in windows:
        start = window["start_index"]
        end = window["end_index_exclusive"]
        window_values = residual_full[start:end]
        if any(value is None for value in window_values):
            continue
        residual_window = [float(value) for value in window_values if value is not None]
        normalized, mean_before, std_before = zscore(residual_window)
        epsilon, estimated_rr = recurrence_epsilon_for_rr(normalized, RECURRENCE_TARGET_RR)
        figure_file = f"phase6_{window['window']}_ar_residual_rp.png"
        achieved_rr = write_recurrence_png(figures_dir / figure_file, normalized, epsilon)
        recurrence_rows.append(
            {
                "series": "ar_residual",
                "window": window["window"],
                "start_time": window["start_time"],
                "end_time": window["end_time"],
                "start_index": start,
                "end_index_exclusive": end,
                "n": len(residual_window),
                "normalization": "z-score por ventana",
                "metric": "absolute_distance_1d",
                "target_rr": RECURRENCE_TARGET_RR,
                "epsilon": epsilon,
                "estimated_rr_from_sorted_values": estimated_rr,
                "achieved_rr": achieved_rr,
                "mean_before_zscore": mean_before,
                "std_before_zscore": std_before,
                "figure_file": figure_file,
            }
        )

    train_info = {
        "train_start": times[0],
        "train_end": times[train_end - 1],
        "train_n": train_end,
        "full_n": len(times),
        "train_mean_log_rv_past_12": train_mean,
        "train_std_log_rv_past_12": train_std,
        "max_ar_order_tested": MAX_AR_ORDER,
        "selected_ar_order_bic": selected_model.order,
        "selected_innovation_variance": selected_model.innovation_variance,
    }

    write_rows_csv(
        tables_dir / "phase6_ar_order_selection.csv",
        ar_rows,
        ["p", "nobs_train", "innovation_variance", "aic", "bic"],
    )
    write_rows_csv(
        tables_dir / "phase6_top10_ar_bic.csv",
        top_models,
        ["p", "nobs_train", "innovation_variance", "aic", "bic"],
    )
    write_rows_csv(
        tables_dir / "phase6_ar_coefficients.csv",
        coeff_rows,
        ["lag", "coefficient", "abs_coefficient", "rank_abs"],
    )
    write_rows_csv(
        tables_dir / "phase6_residual_series.csv",
        residual_table,
        ["open_time", "log_rv_past_12", "z_log_rv_past_12", "fitted_z", "residual"],
    )
    write_rows_csv(
        tables_dir / "phase6_residual_statistics.csv",
        [residual_stats],
        list(residual_stats.keys()),
    )
    write_rows_csv(
        tables_dir / "phase6_residual_acf_values.csv",
        [{"lag": lag, "acf": value} for lag, value in enumerate(residual_acf)],
        ["lag", "acf"],
    )
    write_rows_csv(
        tables_dir / "phase6_residual_pacf_values.csv",
        [{"lag": lag, "pacf": value} for lag, value in enumerate(residual_pacf)],
        ["lag", "pacf"],
    )
    write_rows_csv(
        tables_dir / "phase6_ljung_box_residuals.csv",
        ljung_rows,
        ["lag", "q_stat", "df_adjusted", "p_value", "reject_5pct"],
    )
    write_rows_csv(
        tables_dir / "phase6_correlogram_comparison.csv",
        correlogram_rows,
        [
            "series",
            "nobs",
            "significance_band",
            "acf_lag_1",
            "acf_lag_12",
            "acf_lag_288",
            "acf_lag_2016",
            "acf_significant_lags_1_288",
            "acf_significant_lags_1_2016",
            "pacf_lag_1",
            "pacf_lag_12",
            "pacf_lag_288",
            "pacf_significant_lags_1_288",
        ],
    )
    write_rows_csv(
        tables_dir / "phase6_residual_recurrence_parameters.csv",
        recurrence_rows,
        [
            "series",
            "window",
            "start_time",
            "end_time",
            "start_index",
            "end_index_exclusive",
            "n",
            "normalization",
            "metric",
            "target_rr",
            "epsilon",
            "estimated_rr_from_sorted_values",
            "achieved_rr",
            "mean_before_zscore",
            "std_before_zscore",
            "figure_file",
        ],
    )

    print("FASE 6 - FILTRADO LINEAL")
    print("=" * 72)
    print(f"Train: {times[0]} -> {times[train_end - 1]} ({train_end:,} obs.)")
    print(f"AR seleccionado por BIC: p={selected_model.order}")
    print(f"Tablas: {tables_dir}")
    print(f"Figuras: {figures_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
