"""Fase 9: cuantificacion de la dinamica reconstruida."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from dynamics_quantification import (
    contiguous_block,
    correlation_curve_from_distances,
    pairwise_distances_sorted,
    permutation_entropy,
    radii_from_distances,
    rosenstein_curve,
    select_evenly_spaced_pairs,
    shuffled_scalar,
    summarize_correlation_dimension,
    summarize_lyapunov,
)
from state_space import build_embedding_rows, standardize_train


SERIES_COLUMN = "log_rv_past_12"
TRAIN_END = "2025-06-30 23:55:00"
RANDOM_SEED = 20260602
CORR_SAMPLE_SIZE = 2500
CORR_RADII_COUNT = 32
LYAP_BLOCK_SIZE = 3000
LYAP_K_MAX = 60
LYAP_FIT_START = 1
LYAP_FIT_END = 30
PERMUTATION_ORDERS = [3, 4, 5, 6, 7]
FIVE_MINUTES_PER_HOUR = 12


def read_main_series(path: Path) -> tuple[list[str], list[float]]:
    """Lee open_time y log_rv_past_12."""
    times: list[str] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"open_time", SERIES_COLUMN}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
        for row in reader:
            times.append(row["open_time"])
            values.append(float(row[SERIES_COLUMN]))
    return times, values


def train_end_index(times: list[str], train_end: str) -> int:
    index = 0
    while index < len(times) and times[index] <= train_end:
        index += 1
    if index == 0:
        raise ValueError("No hay observaciones de entrenamiento")
    return index


def load_embedding_params(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}; ejecuta primero la Fase 8")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = build_parser().parse_args()
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = load_embedding_params(args.phase8_params)
    tau = int(params["tau_selected"])
    dim = int(params["m_selected"])
    theiler_window = dim * tau

    times, values = read_main_series(args.input)
    train_end = train_end_index(times, TRAIN_END)
    z_values, train_mean, train_std = standardize_train(values, train_end)
    train_indices = list(range(train_end))
    x_train, train_vector_indices, train_vector_times = build_embedding_rows(
        z_values, times, train_indices, tau, dim
    )
    print(
        f"Fase 9: tau={tau}, m={dim}, train_vectors={len(x_train):,}, "
        f"theiler={theiler_window}"
    )

    shuffled_z_train = shuffled_scalar(z_values[:train_end], RANDOM_SEED)
    shuffled_times = times[:train_end]
    shuffled_indices = list(range(train_end))
    x_shuffled, shuffled_vector_indices, _ = build_embedding_rows(
        shuffled_z_train, shuffled_times, shuffled_indices, tau, dim
    )

    corr_rows, corr_summary = run_correlation_dimension(
        x_train,
        train_vector_indices,
        x_shuffled,
        shuffled_vector_indices,
        theiler_window,
    )

    lyap_rows, lyap_summary = run_lyapunov(
        x_train,
        x_shuffled,
        theiler_window,
    )

    pe_rows = run_permutation_entropy(z_values[:train_end], shuffled_z_train, tau)
    pe_summary = summarize_permutation_entropy(pe_rows)

    quant_summary = {
        "series": "z_log_rv_past_12",
        "tau": tau,
        "m": dim,
        "train_size": train_end,
        "train_vectors": len(x_train),
        "theiler_window": theiler_window,
        "random_seed": RANDOM_SEED,
        "correlation_dimension": corr_summary,
        "lyapunov": lyap_summary,
        "permutation_entropy": pe_summary,
    }

    write_outputs(
        tables_dir,
        figures_dir,
        params,
        tau,
        dim,
        theiler_window,
        train_end,
        len(x_train),
        corr_rows,
        corr_summary,
        lyap_rows,
        lyap_summary,
        pe_rows,
        pe_summary,
        quant_summary,
    )

    original_d2 = corr_summary["original"]["d2_estimate"]
    d2_text = (
        f"D2 aprox original={original_d2:.4g}"
        if isinstance(original_d2, float)
        else "D2 sin meseta clara"
    )
    print(
        "Resumen Fase 9 | "
        f"tau={tau}, m={dim}; {d2_text}; "
        f"Lyap pendiente 5min={lyap_summary['original']['slope_per_5min_step']:.4g}; "
        f"PE delay1 original={pe_summary['delay_1_original_mean']:.4g}, "
        f"barajada={pe_summary['delay_1_shuffled_mean']:.4g}"
    )
    return 0


def run_correlation_dimension(
    x_train: list[list[float]],
    train_indices: list[int],
    x_shuffled: list[list[float]],
    shuffled_indices: list[int],
    theiler_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Grassberger-Procaccia aproximado original vs barajado."""
    print(f"Dimension de correlacion: muestra={CORR_SAMPLE_SIZE}")
    original_vectors, original_indices = select_evenly_spaced_pairs(
        x_train, train_indices, CORR_SAMPLE_SIZE
    )
    shuffled_vectors, shuffled_sample_indices = select_evenly_spaced_pairs(
        x_shuffled, shuffled_indices, CORR_SAMPLE_SIZE
    )
    original_distances = pairwise_distances_sorted(original_vectors, original_indices, theiler_window)
    shuffled_distances = pairwise_distances_sorted(
        shuffled_vectors, shuffled_sample_indices, theiler_window
    )
    radii = radii_from_distances(original_distances, CORR_RADII_COUNT)
    original_rows = correlation_curve_from_distances(
        original_distances,
        radii,
        "original",
        len(original_vectors),
        theiler_window,
    )
    shuffled_rows = correlation_curve_from_distances(
        shuffled_distances,
        radii,
        "shuffled",
        len(shuffled_vectors),
        theiler_window,
    )
    summary = {
        "original": summarize_correlation_dimension(original_rows),
        "shuffled": summarize_correlation_dimension(shuffled_rows),
        "sample_size": len(original_vectors),
        "radii_count": len(radii),
        "metric": "euclidean",
        "theiler_window": theiler_window,
    }
    return original_rows + shuffled_rows, summary


def run_lyapunov(
    x_train: list[list[float]],
    x_shuffled: list[list[float]],
    theiler_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rosenstein original vs barajado."""
    print(f"Lyapunov Rosenstein: bloque continuo={LYAP_BLOCK_SIZE}")
    original_block, original_start = contiguous_block(x_train, LYAP_BLOCK_SIZE, "middle")
    shuffled_block, shuffled_start = contiguous_block(x_shuffled, LYAP_BLOCK_SIZE, "middle")
    original_rows, original_neighbors = rosenstein_curve(
        original_block,
        theiler_window=theiler_window,
        k_max=LYAP_K_MAX,
        series_label="original",
    )
    shuffled_rows, shuffled_neighbors = rosenstein_curve(
        shuffled_block,
        theiler_window=theiler_window,
        k_max=LYAP_K_MAX,
        series_label="shuffled",
    )
    original_summary = summarize_lyapunov(original_rows, LYAP_FIT_START, LYAP_FIT_END)
    shuffled_summary = summarize_lyapunov(shuffled_rows, LYAP_FIT_START, LYAP_FIT_END)
    original_summary.update(
        {
            "block_size": len(original_block),
            "block_start_embedding_index": original_start,
            "nearest_neighbor_pairs": original_neighbors,
            "k_max": LYAP_K_MAX,
        }
    )
    shuffled_summary.update(
        {
            "block_size": len(shuffled_block),
            "block_start_embedding_index": shuffled_start,
            "nearest_neighbor_pairs": shuffled_neighbors,
            "k_max": LYAP_K_MAX,
        }
    )
    return original_rows + shuffled_rows, {"original": original_summary, "shuffled": shuffled_summary}


def run_permutation_entropy(
    z_train: list[float],
    shuffled_z_train: list[float],
    tau: int,
) -> list[dict[str, Any]]:
    """Entropia de permutacion original vs barajada."""
    rows: list[dict[str, Any]] = []
    for label, values in [("original", z_train), ("shuffled", shuffled_z_train)]:
        for delay in [1, tau]:
            for order in PERMUTATION_ORDERS:
                row = permutation_entropy(values, order, delay)
                row.update({"series": label})
                rows.append(row)
    return rows


def summarize_permutation_entropy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_for(series: str, delay: int) -> float:
        values = [
            float(row["normalized_entropy"])
            for row in rows
            if row["series"] == series and int(row["delay"]) == delay
        ]
        return sum(values) / len(values)

    delays = sorted({int(row["delay"]) for row in rows})
    tau_delay = max(delays)
    return {
        "delay_1_original_mean": mean_for("original", 1),
        "delay_1_shuffled_mean": mean_for("shuffled", 1),
        "delay_tau": tau_delay,
        "delay_tau_original_mean": mean_for("original", tau_delay),
        "delay_tau_shuffled_mean": mean_for("shuffled", tau_delay),
        "orders": PERMUTATION_ORDERS,
    }


def write_outputs(
    tables_dir: Path,
    figures_dir: Path,
    phase8_params: dict[str, Any],
    tau: int,
    dim: int,
    theiler_window: int,
    train_size: int,
    train_vectors: int,
    corr_rows: list[dict[str, Any]],
    corr_summary: dict[str, Any],
    lyap_rows: list[dict[str, Any]],
    lyap_summary: dict[str, Any],
    pe_rows: list[dict[str, Any]],
    pe_summary: dict[str, Any],
    quant_summary: dict[str, Any],
) -> None:
    write_rows_csv(
        tables_dir / "phase9_correlation_dimension.csv",
        corr_rows,
        [
            "series",
            "radius",
            "log_radius",
            "correlation_sum",
            "log_correlation_sum",
            "local_slope",
            "n_vectors",
            "n_pairs",
            "metric",
            "theiler_window",
        ],
    )
    (tables_dir / "phase9_correlation_dimension_summary.json").write_text(
        json.dumps(corr_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_rows_csv(
        tables_dir / "phase9_lyapunov_rosenstein.csv",
        lyap_rows,
        ["series", "k", "time_minutes", "mean_log_distance", "n_pairs", "theiler_window"],
    )
    (tables_dir / "phase9_lyapunov_summary.json").write_text(
        json.dumps(lyap_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_rows_csv(
        tables_dir / "phase9_permutation_entropy.csv",
        pe_rows,
        [
            "series",
            "order",
            "delay",
            "permutation_entropy",
            "normalized_entropy",
            "n_patterns",
            "unique_patterns",
            "max_patterns",
        ],
    )
    (tables_dir / "phase9_quantification_summary.json").write_text(
        json.dumps(quant_summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    write_correlation_loglog_svg(
        figures_dir / "phase9_correlation_dimension_loglog.svg",
        corr_rows,
    )
    write_local_slope_svg(
        figures_dir / "phase9_correlation_dimension_local_slope.svg",
        corr_rows,
    )
    write_lyapunov_svg(
        figures_dir / "phase9_lyapunov_rosenstein.svg",
        lyap_rows,
        LYAP_FIT_START,
        LYAP_FIT_END,
    )
    write_permutation_entropy_svg(
        figures_dir / "phase9_permutation_entropy.svg",
        pe_rows,
    )
    write_quantification_comparison_svg(
        figures_dir / "phase9_original_vs_shuffled_quantification.svg",
        corr_summary,
        lyap_summary,
        pe_summary,
    )


def write_correlation_loglog_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    original = [
        (float(row["log_radius"]), float(row["log_correlation_sum"]))
        for row in rows
        if row["series"] == "original" and math.isfinite(float(row["log_correlation_sum"]))
    ]
    shuffled = [
        (float(row["log_radius"]), float(row["log_correlation_sum"]))
        for row in rows
        if row["series"] == "shuffled" and math.isfinite(float(row["log_correlation_sum"]))
    ]
    write_xy_lines_svg(
        path,
        [
            {"label": "original", "points": original, "color": "#5f6b7a"},
            {"label": "barajada", "points": shuffled, "color": "#b45f06"},
        ],
        "Dimension de correlacion: log C(r) vs log r",
        "log r",
        "log C(r)",
    )


def write_local_slope_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    original = [
        (float(row["log_radius"]), float(row["local_slope"]))
        for row in rows
        if row["series"] == "original" and math.isfinite(float(row["local_slope"]))
    ]
    shuffled = [
        (float(row["log_radius"]), float(row["local_slope"]))
        for row in rows
        if row["series"] == "shuffled" and math.isfinite(float(row["local_slope"]))
    ]
    write_xy_lines_svg(
        path,
        [
            {"label": "original", "points": original, "color": "#5f6b7a"},
            {"label": "barajada", "points": shuffled, "color": "#b45f06"},
        ],
        "Pendiente local D2(r)",
        "log r",
        "pendiente local",
    )


def write_lyapunov_svg(
    path: Path,
    rows: list[dict[str, Any]],
    fit_start: int,
    fit_end: int,
) -> None:
    original = [
        (float(row["k"]), float(row["mean_log_distance"]))
        for row in rows
        if row["series"] == "original" and math.isfinite(float(row["mean_log_distance"]))
    ]
    shuffled = [
        (float(row["k"]), float(row["mean_log_distance"]))
        for row in rows
        if row["series"] == "shuffled" and math.isfinite(float(row["mean_log_distance"]))
    ]
    write_xy_lines_svg(
        path,
        [
            {"label": "original", "points": original, "color": "#5f6b7a"},
            {"label": "barajada", "points": shuffled, "color": "#b45f06"},
        ],
        "Rosenstein: divergencia media de vecinos",
        "k pasos de 5 minutos",
        "media log distancia",
        vertical_lines=[(fit_start, f"k={fit_start}"), (fit_end, f"k={fit_end}")],
    )


def write_permutation_entropy_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    points_by_series_delay: list[dict[str, Any]] = []
    colors = {
        ("original", 1): "#5f6b7a",
        ("shuffled", 1): "#b45f06",
    }
    delays = sorted({int(row["delay"]) for row in rows})
    tau_delay = max(delays)
    colors[("original", tau_delay)] = "#8a4f9f"
    colors[("shuffled", tau_delay)] = "#6b8e23"
    styles = {
        ("original", 1): {"dash": "", "marker": "circle"},
        ("original", tau_delay): {"dash": "", "marker": "square"},
        ("shuffled", 1): {"dash": "7,4", "marker": "triangle"},
        ("shuffled", tau_delay): {"dash": "2,4", "marker": "diamond"},
    }
    for series in ["original", "shuffled"]:
        for delay in delays:
            points = [
                (float(row["order"]), float(row["normalized_entropy"]))
                for row in rows
                if row["series"] == series and int(row["delay"]) == delay
            ]
            points_by_series_delay.append(
                {
                    "label": f"{series} delay={delay}",
                    "points": points,
                    "color": colors[(series, delay)],
                    "dash": styles[(series, delay)]["dash"],
                    "marker": styles[(series, delay)]["marker"],
                }
            )
    write_xy_lines_svg(
        path,
        points_by_series_delay,
        "Entropia de permutacion normalizada",
        "orden",
        "entropia normalizada",
    )


def write_quantification_comparison_svg(
    path: Path,
    corr_summary: dict[str, Any],
    lyap_summary: dict[str, Any],
    pe_summary: dict[str, Any],
) -> None:
    bars = [
        ("D2 orig", safe_float(corr_summary["original"].get("d2_estimate")), "#5f6b7a"),
        ("D2 bar", safe_float(corr_summary["shuffled"].get("d2_estimate")), "#b45f06"),
        ("Lyap/h orig", safe_float(lyap_summary["original"].get("slope_per_hour")), "#5f6b7a"),
        ("Lyap/h bar", safe_float(lyap_summary["shuffled"].get("slope_per_hour")), "#b45f06"),
        ("PE d1 orig", safe_float(pe_summary.get("delay_1_original_mean")), "#5f6b7a"),
        ("PE d1 bar", safe_float(pe_summary.get("delay_1_shuffled_mean")), "#b45f06"),
    ]
    write_bar_svg(path, bars, "Resumen original vs barajada")


def write_xy_lines_svg(
    path: Path,
    series: list[dict[str, Any]],
    title: str,
    x_label: str,
    y_label: str,
    vertical_lines: list[tuple[float, str]] | None = None,
    width: int = 1040,
    height: int = 430,
) -> None:
    vertical_lines = vertical_lines or []
    all_points = [point for item in series for point in item["points"]]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = expanded_range(min(x_values), max(x_values), 0.04)
    y_min, y_max = expanded_range(min(y_values), max(y_values), 0.08)
    margin = {"left": 78, "right": 38, "top": 58, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_chart_elements(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    for x_value, label in vertical_lines:
        x = x_coord(x_value)
        elements.append(
            f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" '
            f'y2="{margin["top"] + plot_height:.2f}" stroke="#8a2222" '
            f'stroke-width="1" stroke-dasharray="5,4"/>'
        )
        elements.append(
            f'<text x="{x + 4:.2f}" y="{margin["top"] + 15}" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#8a2222">{esc(label)}</text>'
        )
    for item in series:
        points_text = " ".join(
            f"{x_coord(x):.2f},{y_coord(y):.2f}"
            for x, y in item["points"]
            if math.isfinite(x) and math.isfinite(y)
        )
        dash = item.get("dash", "")
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<polyline points="{points_text}" fill="none" stroke="{item["color"]}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )
        for x, y in item["points"]:
            if math.isfinite(x) and math.isfinite(y):
                elements.append(marker_svg(x_coord(x), y_coord(y), item["color"], item.get("marker", "circle"), 3.0))
    for tick in ticks(x_min, x_max, 6):
        x = x_coord(tick)
        elements.append(
            f'<text x="{x:.2f}" y="{height - 30}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11">{tick:.3g}</text>'
        )
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, x_label, y_label))
    for index, item in enumerate(series[:6]):
        elements.append(
            legend_line_item(
                width - 245,
                28 + 18 * index,
                item["color"],
                item["label"],
                item.get("dash", ""),
                item.get("marker", "circle"),
            )
        )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def write_bar_svg(
    path: Path,
    bars: list[tuple[str, float, str]],
    title: str,
    width: int = 900,
    height: int = 420,
) -> None:
    finite_values = [value for _, value, _ in bars if math.isfinite(value)]
    y_min, y_max = expanded_range(min(0.0, min(finite_values)), max(finite_values), 0.10)
    margin = {"left": 76, "right": 28, "top": 58, "bottom": 92}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_chart_elements(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    zero_y = y_coord(0.0)
    bar_width = plot_width / len(bars) * 0.62
    for index, (label, value, color) in enumerate(bars):
        x = margin["left"] + plot_width * (index + 0.5) / len(bars)
        y = y_coord(value)
        top, bottom = sorted([y, zero_y])
        elements.append(
            f'<rect x="{x - bar_width/2:.2f}" y="{top:.2f}" width="{bar_width:.2f}" '
            f'height="{max(1.0, bottom - top):.2f}" fill="{color}" opacity="0.86"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{height - 58}" text-anchor="end" '
            f'transform="rotate(-35 {x:.2f},{height - 58})" '
            f'font-family="Arial, sans-serif" font-size="10">{esc(label)}</text>'
        )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def base_chart_elements(
    width: int,
    height: int,
    margin: dict[str, int],
    title: str,
    y_min: float,
    y_max: float,
    y_coord: Any,
    plot_width: float,
    plot_height: float,
) -> list[str]:
    elements = [
        svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">{esc(title)}</text>'
        ),
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222222" stroke-width="1"/>',
    ]
    for tick in ticks(y_min, y_max, 5):
        y = y_coord(tick)
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="#e8e8e8" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{tick:.3g}</text>'
        )
    return elements


def axis_labels(
    width: int,
    height: int,
    margin: dict[str, int],
    plot_width: float,
    plot_height: float,
    x_label: str,
    y_label: str,
) -> str:
    return "\n".join(
        [
            f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" '
            f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(x_label)}</text>',
            f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
            f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(y_label)}</text>',
        ]
    )






def rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            rounded[key] = ""
        elif isinstance(value, float):
            rounded[key] = f"{value:.6g}"
        else:
            rounded[key] = value
    return rounded


def safe_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and value is not None else float("nan")


def expanded_range(y_min: float, y_max: float, pad_fraction: float) -> tuple[float, float]:
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        return y_min - delta, y_max + delta
    padding = pad_fraction * (y_max - y_min)
    return y_min - padding, y_max + padding


def ticks(y_min: float, y_max: float, count: int) -> list[float]:
    return [y_min + (y_max - y_min) * index / (count - 1) for index in range(count)]


def legend_item(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<g><circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
        f'<text x="{x + 10:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
        f'font-size="12">{esc(label)}</text></g>'
    )


def legend_line_item(
    x: float,
    y: float,
    color: str,
    label: str,
    dash: str,
    marker: str,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return "\n".join(
        [
            "<g>",
            f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 28:.2f}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="2"{dash_attr}/>',
            marker_svg(x + 14, y, color, marker, 4.0),
            f'<text x="{x + 36:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
            f'font-size="12">{esc(label)}</text>',
            "</g>",
        ]
    )


def marker_svg(x: float, y: float, color: str, marker: str, size: float) -> str:
    if marker == "square":
        return (
            f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{2 * size:.2f}" '
            f'height="{2 * size:.2f}" fill="{color}" opacity="0.9"/>'
        )
    if marker == "triangle":
        points = [
            (x, y - size),
            (x - size * 0.95, y + size * 0.85),
            (x + size * 0.95, y + size * 0.85),
        ]
        return polygon_svg(points, color)
    if marker == "diamond":
        points = [(x, y - size), (x - size, y), (x, y + size), (x + size, y)]
        return polygon_svg(points, color)
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}" opacity="0.9"/>'


def polygon_svg(points: list[tuple[float, float]], color: str) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{point_text}" fill="{color}" opacity="0.9"/>'


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 9: cuantificacion dinamica.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument(
        "--phase8-params",
        type=Path,
        default=Path("reports/tables/phase8_selected_embedding_params.json"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
