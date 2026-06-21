"""Fase 10: contrastes con barajado y datos subrogados."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from state_space import standardize_train
from surrogate_tests import (
    STAT_COLUMNS,
    aaft_surrogate,
    compute_surrogate_statistics,
    phase_randomized,
    summarize_surrogate_group,
)


SERIES_COLUMN = "log_rv_past_12"
TRAIN_END = "2025-06-30 23:55:00"
WINDOW_SIZE = 8192
N_SHUFFLED = 50
N_PHASE_RANDOMIZED = 39
N_AAFT = 39
RANDOM_SEED = 20260602
CORR_SAMPLE_SIZE = 700
CORR_RADII_COUNT = 24
LYAP_BLOCK_SIZE = 1600
LYAP_REFERENCE_COUNT = 550
LYAP_K_MAX = 40
LYAP_FIT_START = 1
LYAP_FIT_END = 20


def read_main_series(path: Path) -> tuple[list[str], list[float]]:
    """Lee open_time y log_rv_past_12."""
    times: list[str] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = float(row[SERIES_COLUMN])
            if math.isfinite(value):
                times.append(row["open_time"])
                values.append(value)
    return times, values


def train_end_index(times: list[str], train_end: str) -> int:
    index = 0
    while index < len(times) and times[index] <= train_end:
        index += 1
    if index == 0:
        raise ValueError("No hay observaciones de entrenamiento")
    return index


def load_phase8_params(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "tau_selected": 137,
        "m_selected": 5,
        "theiler_window": 685,
    }


def main() -> int:
    args = build_parser().parse_args()
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    phase8_params = load_phase8_params(args.phase8_params)
    tau = int(phase8_params.get("tau_selected", 137))
    dim = int(phase8_params.get("m_selected", 5))
    theiler_window = int(phase8_params.get("theiler_window") or tau * dim)
    if theiler_window < tau * dim:
        theiler_window = tau * dim

    times, values = read_main_series(args.input)
    train_end = train_end_index(times, TRAIN_END)
    z_values, train_mean, train_std = standardize_train(values, train_end)
    train_window_start = train_end // 2 - WINDOW_SIZE // 2
    train_window_end = train_window_start + WINDOW_SIZE
    window = z_values[train_window_start:train_window_end]
    window_times = times[train_window_start:train_window_end]
    rng = random.Random(RANDOM_SEED)

    config = {
        "series": "z_log_rv_past_12",
        "input": str(args.input),
        "train_start": times[0],
        "train_end": times[train_end - 1],
        "train_size": train_end,
        "window_size": WINDOW_SIZE,
        "window_start_index": train_window_start,
        "window_end_index_exclusive": train_window_end,
        "window_start_time": window_times[0],
        "window_end_time": window_times[-1],
        "tau": tau,
        "m": dim,
        "theiler_window": theiler_window,
        "n_shuffled": N_SHUFFLED,
        "n_phase_randomized": N_PHASE_RANDOMIZED,
        "n_aaft": N_AAFT,
        "random_seed": RANDOM_SEED,
        "corr_sample_size": CORR_SAMPLE_SIZE,
        "corr_radii_count": CORR_RADII_COUNT,
        "lyap_block_size": LYAP_BLOCK_SIZE,
        "lyap_reference_count": LYAP_REFERENCE_COUNT,
        "lyap_k_max": LYAP_K_MAX,
        "lyap_fit_start": LYAP_FIT_START,
        "lyap_fit_end": LYAP_FIT_END,
        "standardization_mean_train": train_mean,
        "standardization_std_train": train_std,
    }
    print(
        f"Fase 10: ventana {WINDOW_SIZE}, tau={tau}, m={dim}, "
        f"theiler={theiler_window}"
    )

    original_stats = compute_stats_for_series(window, tau, dim, theiler_window)
    print("Original calculada.")

    shuffled_rows = run_replicates(
        "shuffled",
        N_SHUFFLED,
        lambda: shuffled_replicate(window, rng),
        tau,
        dim,
        theiler_window,
    )
    phase_rows = run_replicates(
        "phase_randomized",
        N_PHASE_RANDOMIZED,
        lambda: phase_randomized(window, rng),
        tau,
        dim,
        theiler_window,
    )
    aaft_rows = run_replicates(
        "aaft",
        N_AAFT,
        lambda: aaft_surrogate(window, rng),
        tau,
        dim,
        theiler_window,
    )

    summary_rows = (
        summarize_surrogate_group("shuffled", original_stats, shuffled_rows)
        + summarize_surrogate_group("phase_randomized", original_stats, phase_rows)
        + summarize_surrogate_group("aaft", original_stats, aaft_rows)
    )

    write_outputs(
        tables_dir,
        figures_dir,
        config,
        original_stats,
        shuffled_rows,
        phase_rows,
        aaft_rows,
        summary_rows,
        window,
        phase_rows[0] if phase_rows else {},
        aaft_rows[0] if aaft_rows else {},
    )

    print_summary(original_stats, summary_rows)
    return 0


def compute_stats_for_series(
    values: list[float],
    tau: int,
    dim: int,
    theiler_window: int,
) -> dict[str, float]:
    return compute_surrogate_statistics(
        values,
        tau=tau,
        dim=dim,
        theiler_window=theiler_window,
        corr_sample_size=CORR_SAMPLE_SIZE,
        corr_radii_count=CORR_RADII_COUNT,
        lyap_block_size=LYAP_BLOCK_SIZE,
        lyap_reference_count=LYAP_REFERENCE_COUNT,
        lyap_k_max=LYAP_K_MAX,
        lyap_fit_start=LYAP_FIT_START,
        lyap_fit_end=LYAP_FIT_END,
    )


def shuffled_replicate(values: list[float], rng: random.Random) -> list[float]:
    replicate = values[:]
    rng.shuffle(replicate)
    return replicate


def run_replicates(
    group: str,
    count: int,
    generator: Any,
    tau: int,
    dim: int,
    theiler_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate in range(1, count + 1):
        values = generator()
        stats = compute_stats_for_series(values, tau, dim, theiler_window)
        row: dict[str, Any] = {"group": group, "replicate": replicate}
        row.update(stats)
        rows.append(row)
        if replicate % 10 == 0 or replicate == count:
            print(f"{group}: {replicate}/{count}")
    return rows


def write_outputs(
    tables_dir: Path,
    figures_dir: Path,
    config: dict[str, Any],
    original_stats: dict[str, float],
    shuffled_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    aaft_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    original_window: list[float],
    phase_example_stats: dict[str, Any],
    aaft_example_stats: dict[str, Any],
) -> None:
    (tables_dir / "phase10_original_stats.json").write_text(
        json.dumps(clean_json(original_stats), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (tables_dir / "phase10_config.json").write_text(
        json.dumps(clean_json(config), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_rows_csv(tables_dir / "phase10_shuffled_stats.csv", shuffled_rows, ["group", "replicate", *STAT_COLUMNS])
    write_rows_csv(
        tables_dir / "phase10_phase_randomized_stats.csv",
        phase_rows,
        ["group", "replicate", *STAT_COLUMNS],
    )
    write_rows_csv(tables_dir / "phase10_aaft_stats.csv", aaft_rows, ["group", "replicate", *STAT_COLUMNS])
    write_rows_csv(
        tables_dir / "phase10_surrogate_summary.csv",
        summary_rows,
        [
            "group",
            "statistic",
            "original",
            "mean",
            "std",
            "min",
            "p05",
            "median",
            "p95",
            "max",
            "S",
            "empirical_p_value",
            "n_success",
            "n_failures",
        ],
    )

    write_boxplots_svg(figures_dir / "phase10_surrogate_boxplots.svg", original_stats, summary_rows)
    write_separated_boxplot_svgs(figures_dir, original_stats, summary_rows)
    write_hist_svg(
        figures_dir / "phase10_lyapunov_surrogate_hist.svg",
        original_stats,
        [shuffled_rows, phase_rows, aaft_rows],
        "lyapunov_slope_per_hour",
        "Lyapunov aproximado por hora: original vs contrastes",
    )
    write_hist_svg(
        figures_dir / "phase10_permutation_entropy_surrogate_hist.svg",
        original_stats,
        [shuffled_rows, phase_rows, aaft_rows],
        "permutation_entropy_delay_1",
        "Entropia de permutacion delay=1: original vs contrastes",
    )
    write_example_surrogates_svg(
        figures_dir / "phase10_example_surrogates.svg",
        original_window,
        shuffled_replicate(original_window, random.Random(RANDOM_SEED + 101)),
        phase_randomized(original_window, random.Random(RANDOM_SEED + 102)),
        aaft_surrogate(original_window, random.Random(RANDOM_SEED + 103)),
    )


def write_boxplots_svg(
    path: Path,
    original_stats: dict[str, float],
    summary_rows: list[dict[str, Any]],
    width: int = 1180,
    height: int = 560,
) -> None:
    groups = ["shuffled", "phase_randomized", "aaft"]
    colors = {"shuffled": "#6f7f8f", "phase_randomized": "#b45f06", "aaft": "#6b8e23"}
    rows_by_key = {(row["group"], row["statistic"]): row for row in summary_rows}
    finite_values = []
    for row in summary_rows:
        if math.isfinite(float(row["p05"])):
            finite_values.extend([float(row["p05"]), float(row["p95"]), float(row["original"])])
    y_min, y_max = expanded_range(min(finite_values), max(finite_values), 0.08)
    margin = {"left": 76, "right": 28, "top": 58, "bottom": 138}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_chart_elements(width, height, margin, "Resumen de subrogadas por estadistico", y_min, y_max, y_coord, plot_width, plot_height)
    slot_count = len(STAT_COLUMNS)
    for stat_index, stat in enumerate(STAT_COLUMNS):
        center_x = margin["left"] + plot_width * (stat_index + 0.5) / slot_count
        group_width = plot_width / slot_count / 4.8
        for group_index, group in enumerate(groups):
            row = rows_by_key[(group, stat)]
            x = center_x + (group_index - 1) * group_width
            p05 = float(row["p05"])
            p95 = float(row["p95"])
            median = float(row["median"])
            lower = y_coord(p95)
            upper = y_coord(p05)
            elements.append(
                f'<rect x="{x - group_width*0.34:.2f}" y="{lower:.2f}" width="{group_width*0.68:.2f}" '
                f'height="{max(1.0, upper - lower):.2f}" fill="{colors[group]}" opacity="0.42"/>'
            )
            elements.append(
                f'<line x1="{x - group_width*0.40:.2f}" y1="{y_coord(median):.2f}" '
                f'x2="{x + group_width*0.40:.2f}" y2="{y_coord(median):.2f}" '
                f'stroke="{colors[group]}" stroke-width="2"/>'
            )
        original = float(original_stats[stat])
        elements.append(
            f'<line x1="{center_x - group_width*1.75:.2f}" y1="{y_coord(original):.2f}" '
            f'x2="{center_x + group_width*1.75:.2f}" y2="{y_coord(original):.2f}" '
            f'stroke="#8a2222" stroke-width="2.2" stroke-dasharray="4,3"/>'
        )
        elements.append(
            f'<text x="{center_x:.2f}" y="{height - 76}" text-anchor="end" '
            f'transform="rotate(-35 {center_x:.2f},{height - 76})" '
            f'font-family="Arial, sans-serif" font-size="10">{esc(stat)}</text>'
        )
    elements.append(legend_item(width - 245, 28, "#6f7f8f", "barajadas p05-p95"))
    elements.append(legend_item(width - 245, 46, "#b45f06", "phase randomized"))
    elements.append(legend_item(width - 245, 64, "#6b8e23", "AAFT"))
    elements.append(
        f'<line x1="{width - 245:.2f}" y1="82" x2="{width - 218:.2f}" y2="82" '
        f'stroke="#8a2222" stroke-width="2.2" stroke-dasharray="4,3"/>'
    )
    elements.append(
        f'<text x="{width - 209:.2f}" y="86" font-family="Arial, sans-serif" font-size="12">original</text>'
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_separated_boxplot_svgs(
    figures_dir: Path,
    original_stats: dict[str, float],
    summary_rows: list[dict[str, Any]],
) -> None:
    write_stat_group_boxplot_svg(
        figures_dir / "phase10_surrogate_boxplots_d2.svg",
        original_stats,
        summary_rows,
        ["d2"],
        "Subrogadas: dimension de correlacion D2",
        "D2 aproximada",
    )
    write_stat_group_boxplot_svg(
        figures_dir / "phase10_surrogate_boxplots_lyapunov.svg",
        original_stats,
        summary_rows,
        ["lyapunov_slope_per_hour"],
        "Subrogadas: maximo exponente de Lyapunov aproximado",
        "pendiente por hora",
    )
    write_stat_group_boxplot_svg(
        figures_dir / "phase10_surrogate_boxplots_entropy.svg",
        original_stats,
        summary_rows,
        ["permutation_entropy_delay_1", "permutation_entropy_delay_tau"],
        "Subrogadas: entropia de permutacion",
        "entropia normalizada",
    )


def write_stat_group_boxplot_svg(
    path: Path,
    original_stats: dict[str, float],
    summary_rows: list[dict[str, Any]],
    stats: list[str],
    title: str,
    y_label: str,
    width: int = 980,
    height: int = 500,
) -> None:
    groups = ["shuffled", "phase_randomized", "aaft"]
    colors = {"shuffled": "#6f7f8f", "phase_randomized": "#b45f06", "aaft": "#6b8e23"}
    rows_by_key = {(row["group"], row["statistic"]): row for row in summary_rows}
    finite_values = []
    for stat in stats:
        finite_values.append(float(original_stats[stat]))
        for group in groups:
            row = rows_by_key[(group, stat)]
            finite_values.extend([float(row["p05"]), float(row["median"]), float(row["p95"])])
    y_min, y_max = expanded_range(min(finite_values), max(finite_values), 0.10)
    margin = {"left": 82, "right": 30, "top": 58, "bottom": 118}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_chart_elements(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    slot_count = len(stats)
    for stat_index, stat in enumerate(stats):
        center_x = margin["left"] + plot_width * (stat_index + 0.5) / slot_count
        slot_width = plot_width / slot_count
        group_width = min(90.0, slot_width / 4.2)
        for group_index, group in enumerate(groups):
            row = rows_by_key[(group, stat)]
            x = center_x + (group_index - 1) * group_width
            p05 = float(row["p05"])
            p95 = float(row["p95"])
            median = float(row["median"])
            lower = y_coord(p95)
            upper = y_coord(p05)
            box_width = group_width * 0.62
            elements.append(
                f'<rect x="{x - box_width/2:.2f}" y="{lower:.2f}" width="{box_width:.2f}" '
                f'height="{max(1.0, upper - lower):.2f}" fill="{colors[group]}" opacity="0.42"/>'
            )
            elements.append(
                f'<line x1="{x - box_width*0.58:.2f}" y1="{y_coord(median):.2f}" '
                f'x2="{x + box_width*0.58:.2f}" y2="{y_coord(median):.2f}" '
                f'stroke="{colors[group]}" stroke-width="2"/>'
            )
        original = float(original_stats[stat])
        elements.append(
            f'<line x1="{center_x - group_width*1.85:.2f}" y1="{y_coord(original):.2f}" '
            f'x2="{center_x + group_width*1.85:.2f}" y2="{y_coord(original):.2f}" '
            f'stroke="#8a2222" stroke-width="2.2" stroke-dasharray="4,3"/>'
        )
        label_y = height - 72
        elements.append(
            f'<text x="{center_x:.2f}" y="{label_y}" text-anchor="end" '
            f'transform="rotate(-28 {center_x:.2f},{label_y})" '
            f'font-family="Arial, sans-serif" font-size="11">{esc(stat)}</text>'
        )
    elements.append(legend_item(width - 232, 28, "#6f7f8f", "barajadas p05-p95"))
    elements.append(legend_item(width - 232, 46, "#b45f06", "phase randomized"))
    elements.append(legend_item(width - 232, 64, "#6b8e23", "AAFT"))
    elements.append(legend_line(width - 232, 82, "#8a2222", "original"))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "estadistico", y_label))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_hist_svg(
    path: Path,
    original_stats: dict[str, float],
    group_rows: list[list[dict[str, Any]]],
    stat: str,
    title: str,
    width: int = 980,
    height: int = 430,
) -> None:
    colors = ["#6f7f8f", "#b45f06", "#6b8e23"]
    labels = ["barajadas", "phase", "AAFT"]
    values_by_group = [
        [float(row[stat]) for row in rows if math.isfinite(float(row[stat]))]
        for rows in group_rows
    ]
    all_values = [value for values in values_by_group for value in values]
    original = float(original_stats[stat])
    x_min, x_max = expanded_range(min(all_values + [original]), max(all_values + [original]), 0.08)
    bins = 24
    histograms = [histogram_counts(values, x_min, x_max, bins) for values in values_by_group]
    y_max = max(max(counts) for counts in histograms)
    margin = {"left": 72, "right": 30, "top": 58, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * value / max(1, y_max)

    elements = base_chart_elements(width, height, margin, title, 0.0, y_max, y_coord, plot_width, plot_height)
    bin_width = plot_width / bins
    for group_index, counts in enumerate(histograms):
        for bin_index, count in enumerate(counts):
            x = margin["left"] + bin_index * bin_width + group_index * bin_width / 4.0
            y = y_coord(count)
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bin_width/4.5:.2f}" '
                f'height="{plot_height + margin["top"] - y:.2f}" fill="{colors[group_index]}" '
                f'opacity="0.65"/>'
            )
    x_original = x_coord(original)
    elements.append(
        f'<line x1="{x_original:.2f}" y1="{margin["top"]}" x2="{x_original:.2f}" '
        f'y2="{margin["top"] + plot_height:.2f}" stroke="#8a2222" stroke-width="2" '
        f'stroke-dasharray="4,3"/>'
    )
    for index, label in enumerate(labels):
        elements.append(legend_item(width - 170, 28 + 18 * index, colors[index], label))
    elements.append(legend_line(width - 170, 82, "#8a2222", "original"))
    elements.extend(x_axis_ticks(margin, plot_width, plot_height, x_min, x_max, x_coord, count=6))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, statistic_axis_label(stat), "frecuencia"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_example_surrogates_svg(
    path: Path,
    original: list[float],
    shuffled: list[float],
    phase: list[float],
    aaft: list[float],
    width: int = 1120,
    height: int = 520,
) -> None:
    series = [
        ("original", original[:512], "#5f6b7a"),
        ("barajada", shuffled[:512], "#6f7f8f"),
        ("phase", phase[:512], "#b45f06"),
        ("AAFT", aaft[:512], "#6b8e23"),
    ]
    all_values = [value for _, values, _ in series for value in values]
    y_min, y_max = expanded_range(min(all_values), max(all_values), 0.08)
    margin = {"left": 96, "right": 30, "top": 58, "bottom": 70}
    plot_width = width - margin["left"] - margin["right"]
    row_height = (height - margin["top"] - margin["bottom"]) / len(series)

    def x_coord(index: int) -> float:
        return margin["left"] + plot_width * index / 511

    elements = [
        svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="18" font-weight="700">Ejemplos de serie original y subrogadas</text>',
    ]
    for row_index, (label, values, color) in enumerate(series):
        top = margin["top"] + row_index * row_height
        bottom = top + row_height - 12
        mid = (top + bottom) / 2

        def y_coord(value: float) -> float:
            return bottom - (bottom - top) * (value - y_min) / (y_max - y_min)

        for tick in ticks(y_min, y_max, 3):
            y = y_coord(tick)
            elements.append(f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" y2="{y:.2f}" stroke="#eeeeee"/>')
            elements.append(f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10">{tick:.3g}</text>')
        points = " ".join(f"{x_coord(i):.2f},{y_coord(value):.2f}" for i, value in enumerate(values))
        elements.append(
            f'<text x="{margin["left"] - 10}" y="{mid + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12">{esc(label)}</text>'
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.2"/>'
        )
        if row_index == len(series) - 1:
            for position in [0, 128, 256, 384, 511]:
                x = x_coord(position)
                elements.append(f'<line x1="{x:.2f}" y1="{bottom:.2f}" x2="{x:.2f}" y2="{bottom + 5:.2f}" stroke="#222222"/>')
                elements.append(f'<text x="{x:.2f}" y="{bottom + 18:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{position}</text>')
    elements.append(axis_labels(width, height, margin, plot_width, height - margin["top"] - margin["bottom"], "indice en ventana (velas de 5 min)", "z_log_rv_past_12"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")




def print_summary(original_stats: dict[str, float], summary_rows: list[dict[str, Any]]) -> None:
    lookup = {(row["group"], row["statistic"]): row for row in summary_rows}
    print("Resumen Fase 10")
    print(f"Original d2={original_stats['d2']:.4g}")
    print(f"Original lyap/h={original_stats['lyapunov_slope_per_hour']:.4g}")
    print(f"Original PE d1={original_stats['permutation_entropy_delay_1']:.4g}")
    for group in ["shuffled", "phase_randomized", "aaft"]:
        row = lookup[(group, "permutation_entropy_delay_1")]
        print(
            f"{group}: PE d1 mean={float(row['mean']):.4g}, "
            f"S={float(row['S']):.4g}, p={float(row['empirical_p_value']):.4g}"
        )
    print("Contraste completado.")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)
    return grouped


def summary_columns() -> list[str]:
    return [
        "group",
        "statistic",
        "original",
        "mean",
        "std",
        "p05",
        "median",
        "p95",
        "S",
        "empirical_p_value",
        "n_success",
        "n_failures",
    ]


def histogram_counts(values: list[float], lower: float, upper: float, bins: int) -> list[int]:
    counts = [0] * bins
    for value in values:
        index = int((value - lower) / (upper - lower) * bins)
        index = max(0, min(bins - 1, index))
        counts[index] += 1
    return counts


def expanded_range(y_min: float, y_max: float, pad_fraction: float) -> tuple[float, float]:
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        return y_min - delta, y_max + delta
    padding = pad_fraction * (y_max - y_min)
    return y_min - padding, y_max + padding


def ticks(y_min: float, y_max: float, count: int) -> list[float]:
    return [y_min + (y_max - y_min) * index / (count - 1) for index in range(count)]


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
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="18" font-weight="700">{esc(title)}</text>',
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


def x_axis_ticks(
    margin: dict[str, int],
    plot_width: float,
    plot_height: float,
    x_min: float,
    x_max: float,
    x_coord: Any,
    count: int = 6,
) -> list[str]:
    elements: list[str] = []
    bottom = margin["top"] + plot_height
    for tick in ticks(x_min, x_max, count):
        x = x_coord(tick)
        elements.append(f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" y2="{bottom:.2f}" stroke="#f0f0f0"/>')
        elements.append(f'<line x1="{x:.2f}" y1="{bottom:.2f}" x2="{x:.2f}" y2="{bottom + 5:.2f}" stroke="#222222"/>')
        elements.append(f'<text x="{x:.2f}" y="{bottom + 18:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{tick:.3g}</text>')
    return elements


def statistic_axis_label(stat: str) -> str:
    labels = {
        "lyapunov_slope_per_hour": "exponente de Lyapunov estimado por hora",
        "permutation_entropy_delay_1": "entropia de permutacion normalizada (delay=1)",
        "permutation_entropy_delay_tau": "entropia de permutacion normalizada (delay=tau)",
        "lyapunov_slope_per_step": "exponente de Lyapunov estimado por paso",
        "d2": "dimension de correlacion estimada",
    }
    return labels.get(stat, stat)


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


def legend_item(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<g><circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
        f'<text x="{x + 10:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
        f'font-size="12">{esc(label)}</text></g>'
    )


def legend_line(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<g><line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 28:.2f}" y2="{y:.2f}" '
        f'stroke="{color}" stroke-width="2" stroke-dasharray="4,3"/>'
        f'<text x="{x + 36:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
        f'font-size="12">{esc(label)}</text></g>'
    )




def rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            rounded[key] = ""
        elif isinstance(value, float):
            rounded[key] = f"{value:.6g}" if math.isfinite(value) else "nan"
        else:
            rounded[key] = value
    return rounded


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 10: subrogados.")
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
