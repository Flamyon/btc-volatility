"""Fase 8: reconstruccion del espacio de estados."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from state_space import (
    average_mutual_information_from_bins,
    build_embedding_rows,
    discretize,
    fnn_and_cao,
    quantile_edges,
    rows_around_tau,
    sample_embedding_rows,
    select_final_m,
    select_m_from_cao_unity_plateau,
    select_m_from_fnn,
    select_tau_from_ami,
    shuffled_copy,
    standardize_train,
    write_npz_embedding,
)


SERIES_COLUMN = "log_rv_past_12"
TRAIN_END = "2025-06-30 23:55:00"
AMI_MAX_LAG = 288
AMI_BINS = 32
MAX_DIM = 15
FNN_SAMPLE_SIZE = 3000
CAO_SAMPLE_SIZE = 3000
THEILER_MINIMUM = 12
FNN_RTOL = 10.0
FNN_ATOL = 2.0
FNN_THRESHOLD_PERCENT = 5.0
CAO_UNITY_PLATEAU_TOLERANCE = 0.10
RANDOM_SEED = 20260602
GLOBAL_SCATTER_POINTS = 12000
WINDOW_SCATTER_POINTS = 3500
EMBEDDING_SAMPLE_ROWS_PER_SPLIT = 2500


def read_main_series(path: Path) -> tuple[list[str], list[float], list[float | None]]:
    """Lee open_time, log_rv_past_12 y log_rv_future_12 si existe."""
    times: list[str] = []
    values: list[float] = []
    future_12: list[float | None] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"open_time", SERIES_COLUMN}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
        has_future = "log_rv_future_12" in (reader.fieldnames or [])
        for row in reader:
            times.append(row["open_time"])
            values.append(float(row[SERIES_COLUMN]))
            future_12.append(float(row["log_rv_future_12"]) if has_future else None)
    return times, values, future_12


def train_end_index(times: list[str], train_end: str) -> int:
    """Indice exclusivo del tramo de entrenamiento."""
    index = 0
    while index < len(times) and times[index] <= train_end:
        index += 1
    if index == 0:
        raise ValueError("No hay observaciones de entrenamiento")
    return index


def read_phase5_windows(path: Path) -> list[dict[str, Any]]:
    """Lee ventanas representativas de Fase 5."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            parsed["start_index"] = int(row["start_index"])
            parsed["end_index_exclusive"] = int(row["end_index_exclusive"])
            parsed["n"] = int(row["n"])
            rows.append(parsed)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    processed_dir = args.processed_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    times, values, future_12 = read_main_series(args.input)
    train_end = train_end_index(times, TRAIN_END)
    z_values, train_mean, train_std = standardize_train(values, train_end)
    z_train = z_values[:train_end]
    print(f"Serie cargada: {len(values):,} observaciones; train={train_end:,}")

    edges = quantile_edges(z_train, AMI_BINS)
    binned_train = discretize(z_train, edges)
    ami_rows = average_mutual_information_from_bins(binned_train, AMI_MAX_LAG, AMI_BINS)
    tau_selected, tau_rule = select_tau_from_ami(ami_rows)
    shuffled_bins = shuffled_copy(binned_train, RANDOM_SEED)
    ami_shuffled_rows = average_mutual_information_from_bins(shuffled_bins, AMI_MAX_LAG, AMI_BINS)
    print(f"Tau seleccionado: {tau_selected} ({tau_rule})")

    theiler_window = max(tau_selected, THEILER_MINIMUM)
    fnn_rows, cao_rows, nonlinear_sample_size = fnn_and_cao(
        z_values,
        train_end_index_exclusive=train_end,
        tau=tau_selected,
        max_dim=MAX_DIM,
        sample_size=FNN_SAMPLE_SIZE,
        theiler_window=theiler_window,
        rtol=FNN_RTOL,
        atol=FNN_ATOL,
    )
    m_fnn, m_fnn_rule = select_m_from_fnn(fnn_rows, FNN_THRESHOLD_PERCENT)
    m_cao, m_cao_rule = select_m_from_cao_unity_plateau(
        cao_rows,
        tolerance=CAO_UNITY_PLATEAU_TOLERANCE,
    )
    m_selected, m_rule = select_final_m(m_fnn, m_cao, MAX_DIM)
    print(f"m_FNN={m_fnn}; m_Cao={m_cao}; m_final={m_selected}")

    train_indices = list(range(0, train_end))
    test_indices = list(range(train_end, len(values)))
    x_train, train_vector_indices, train_vector_times = build_embedding_rows(
        z_values, times, train_indices, tau_selected, m_selected
    )
    x_test, test_vector_indices, test_vector_times = build_embedding_rows(
        z_values, times, test_indices, tau_selected, m_selected
    )

    selected_params = {
        "series": "z_log_rv_past_12",
        "source_series": SERIES_COLUMN,
        "train_start": times[0],
        "train_end": times[train_end - 1],
        "test_start": times[train_end] if train_end < len(times) else "",
        "test_end": times[-1],
        "tau_selected": tau_selected,
        "tau_selection_rule": tau_rule,
        "m_fnn": m_fnn,
        "m_fnn_rule": m_fnn_rule,
        "m_cao": m_cao,
        "m_cao_rule": m_cao_rule,
        "m_selected": m_selected,
        "m_selection_rule": m_rule,
        "train_size": train_end,
        "test_size": len(values) - train_end,
        "sample_size_ami": train_end,
        "sample_size_fnn": nonlinear_sample_size,
        "sample_size_cao": nonlinear_sample_size,
        "ami_bins": AMI_BINS,
        "ami_max_lag": AMI_MAX_LAG,
        "max_dim": MAX_DIM,
        "theiler_window": theiler_window,
        "fnn_rtol": FNN_RTOL,
        "fnn_atol": FNN_ATOL,
        "cao_unity_plateau_tolerance": CAO_UNITY_PLATEAU_TOLERANCE,
        "random_seed": RANDOM_SEED,
        "embedding_convention": "X_t=[z_t,z_{t-tau},z_{t-2tau},...,z_{t-(m-1)tau}]",
        "train_embedding_vectors": len(x_train),
        "test_embedding_vectors": len(x_test),
    }

    write_npz_embedding(
        processed_dir / "phase8_embedding_train.npz",
        x_train,
        train_vector_indices,
        selected_params | {"split": "train"},
    )
    write_npz_embedding(
        processed_dir / "phase8_embedding_test.npz",
        x_test,
        test_vector_indices,
        selected_params | {"split": "test"},
    )

    (tables_dir / "phase8_selected_embedding_params.json").write_text(
        json.dumps(selected_params, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    write_outputs(
        tables_dir,
        figures_dir,
        processed_dir,
        times,
        z_values,
        train_end,
        ami_rows,
        ami_shuffled_rows,
        tau_selected,
        fnn_rows,
        cao_rows,
        selected_params,
        x_train,
        train_vector_indices,
        train_vector_times,
        x_test,
        test_vector_indices,
        test_vector_times,
        read_phase5_windows(tables_dir / "phase5_selected_windows.csv"),
    )

    return 0


def write_outputs(
    tables_dir: Path,
    figures_dir: Path,
    processed_dir: Path,
    times: list[str],
    z_values: list[float],
    train_end: int,
    ami_rows: list[dict[str, Any]],
    ami_shuffled_rows: list[dict[str, Any]],
    tau_selected: int,
    fnn_rows: list[dict[str, Any]],
    cao_rows: list[dict[str, Any]],
    selected_params: dict[str, Any],
    x_train: list[list[float]],
    train_vector_indices: list[int],
    train_vector_times: list[str],
    x_test: list[list[float]],
    test_vector_indices: list[int],
    test_vector_times: list[str],
    windows: list[dict[str, Any]],
) -> None:
    """Guarda las tablas y figuras de la reconstruccion."""
    write_rows_csv(
        tables_dir / "phase8_ami_tau.csv",
        ami_rows,
        ["tau", "mutual_information", "n_pairs"],
    )
    write_rows_csv(
        tables_dir / "phase8_ami_shuffled_tau.csv",
        ami_shuffled_rows,
        ["tau", "mutual_information", "n_pairs"],
    )
    write_rows_csv(
        tables_dir / "phase8_fnn.csv",
        fnn_rows,
        ["m", "fnn_fraction", "fnn_percent", "n_used", "false_neighbors", "theiler_window", "rtol", "atol"],
    )
    write_rows_csv(
        tables_dir / "phase8_cao.csv",
        cao_rows,
        ["m", "E1", "E2", "E_m", "E_star_m", "n_used"],
    )
    sample_rows = sample_embedding_rows(
        x_train,
        train_vector_indices,
        train_vector_times,
        "train",
        EMBEDDING_SAMPLE_ROWS_PER_SPLIT,
    ) + sample_embedding_rows(
        x_test,
        test_vector_indices,
        test_vector_times,
        "test",
        EMBEDDING_SAMPLE_ROWS_PER_SPLIT,
    )
    sample_columns = ["split", "open_time", "index"] + [
        f"x{coord}" for coord in range(1, selected_params["m_selected"] + 1)
    ]
    write_rows_csv(tables_dir / "phase8_embedding_sample.csv", sample_rows, sample_columns)

    write_ami_svg(
        figures_dir / "phase8_ami_tau.svg",
        ami_rows,
        selected_params["tau_selected"],
        "Informacion mutua media para seleccionar tau",
    )
    write_ami_comparison_svg(
        figures_dir / "phase8_ami_original_vs_shuffled.svg",
        ami_rows,
        ami_shuffled_rows,
        selected_params["tau_selected"],
    )
    write_fnn_svg(
        figures_dir / "phase8_fnn.svg",
        fnn_rows,
        selected_params["m_fnn"],
        FNN_THRESHOLD_PERCENT,
    )
    write_cao_svg(
        figures_dir / "phase8_cao.svg",
        cao_rows,
        selected_params["m_cao"],
    )
    write_embedding_2d_svg(
        figures_dir / "phase8_embedding_2d.svg",
        x_train,
        train_vector_indices,
        x_test,
        test_vector_indices,
        title=f"Reconstruccion 2D global: tau={selected_params['tau_selected']}, m={selected_params['m_selected']}",
    )
    if selected_params["m_selected"] >= 3:
        write_embedding_3d_svg(
            figures_dir / "phase8_embedding_3d.svg",
            x_train,
            train_vector_indices,
            x_test,
            test_vector_indices,
            title=f"Proyeccion 3D del embedding: tau={selected_params['tau_selected']}, m={selected_params['m_selected']}",
        )

    window_figures = write_window_embedding_figures(
        figures_dir,
        windows,
        z_values,
        times,
        selected_params["tau_selected"],
    )



def write_window_embedding_figures(
    figures_dir: Path,
    windows: list[dict[str, Any]],
    z_values: list[float],
    times: list[str],
    tau: int,
) -> list[dict[str, Any]]:
    """Reconstrucciones 2D por ventanas representativas."""
    output_rows: list[dict[str, Any]] = []
    for window in windows:
        start = int(window["start_index"])
        end = int(window["end_index_exclusive"])
        indices = list(range(start + tau, end))
        vectors, vector_indices, _ = build_embedding_rows(z_values, times, indices, tau, 2)
        if not vectors:
            continue
        figure_file = f"phase8_{window['window']}_embedding_2d.svg"
        write_single_embedding_2d_svg(
            figures_dir / figure_file,
            vectors,
            vector_indices,
            title=f"Embedding 2D - ventana {window['window']} (tau={tau})",
            color="#6f7f8f",
        )
        output_rows.append(
            {
                "window": window["window"],
                "start_time": window["start_time"],
                "end_time": window["end_time"],
                "n_vectors": len(vectors),
                "figure_file": figure_file,
            }
        )
    return output_rows


def write_ami_svg(path: Path, rows: list[dict[str, Any]], selected_tau: int, title: str) -> None:
    xs = [int(row["tau"]) for row in rows]
    ys = [float(row["mutual_information"]) for row in rows]
    write_line_chart_svg(
        path,
        xs,
        [{"label": "AMI", "values": ys, "color": "#5f6b7a"}],
        title,
        "tau",
        "informacion mutua",
        vertical_lines=[(selected_tau, f"tau={selected_tau}")],
        y_limits=(0.0, 1.08 * max(ys)),
    )


def write_ami_comparison_svg(
    path: Path,
    original_rows: list[dict[str, Any]],
    shuffled_rows: list[dict[str, Any]],
    selected_tau: int,
) -> None:
    xs = [int(row["tau"]) for row in original_rows]
    original = [float(row["mutual_information"]) for row in original_rows]
    shuffled = [float(row["mutual_information"]) for row in shuffled_rows]
    selected_index = xs.index(selected_tau)

    width, height = 1040, 460
    margin = {"left": 78, "right": 38, "top": 58, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, 1.08 * max(original + shuffled)

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = [
        svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
            "AMI original vs serie barajada</text>"
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
            f'font-family="Arial, sans-serif" font-size="11">{tick:.4g}</text>'
        )

    selected_x = x_coord(selected_tau)
    elements.append(
        f'<line x1="{selected_x:.2f}" y1="{margin["top"]}" x2="{selected_x:.2f}" '
        f'y2="{margin["top"] + plot_height:.2f}" stroke="#8a2222" '
        f'stroke-width="1" stroke-dasharray="5,4"/>'
    )
    elements.append(
        f'<text x="{selected_x + 4:.2f}" y="{margin["top"] + 15}" '
        f'font-family="Arial, sans-serif" font-size="11" fill="#8a2222">'
        f'primer minimo local: tau={selected_tau}</text>'
    )

    for values, color in [(original, "#5f6b7a"), (shuffled, "#b45f06")]:
        points = " ".join(
            f"{x_coord(x):.2f},{y_coord(y):.2f}"
            for x, y in zip(xs, values)
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in zip(xs, values):
            elements.append(
                f'<circle cx="{x_coord(x):.2f}" cy="{y_coord(y):.2f}" r="1.7" '
                f'fill="{color}" opacity="0.82"/>'
            )
    elements.append(
        f'<circle cx="{selected_x:.2f}" cy="{y_coord(original[selected_index]):.2f}" '
        f'r="4" fill="#8a2222" stroke="#ffffff" stroke-width="1.2"/>'
    )

    for tick in x_ticks(x_min, x_max):
        x = x_coord(tick)
        elements.append(
            f'<text x="{x:.2f}" y="{height - 30}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11">{tick:g}</text>'
        )
    elements.append(
        f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">tau</text>'
    )
    elements.append(
        f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">'
        "informacion mutua</text>"
    )
    elements.append(legend_item(width - 185, 28, "#5f6b7a", "original"))
    elements.append(legend_item(width - 185, 46, "#b45f06", "barajada"))

    zoom_start = max(min(xs), selected_tau - 5)
    zoom_end = min(max(xs), selected_tau + 5)
    zoom_rows = [
        (tau, value)
        for tau, value in zip(xs, original)
        if zoom_start <= tau <= zoom_end
    ]
    zoom_values = [value for _, value in zoom_rows]
    zoom_y_min, zoom_y_max = expanded_range(min(zoom_values), max(zoom_values))
    inset = {"left": 640.0, "top": 112.0, "width": 330.0, "height": 175.0}

    def inset_x(value: float) -> float:
        return inset["left"] + inset["width"] * (value - zoom_start) / (zoom_end - zoom_start)

    def inset_y(value: float) -> float:
        return (
            inset["top"]
            + inset["height"]
            - inset["height"] * (value - zoom_y_min) / (zoom_y_max - zoom_y_min)
        )

    elements.append(
        f'<rect x="{inset["left"]}" y="{inset["top"]}" width="{inset["width"]}" '
        f'height="{inset["height"]}" fill="#ffffff" fill-opacity="0.96" '
        f'stroke="#555555" stroke-width="1"/>'
    )
    elements.append(
        f'<text x="{inset["left"] + 8:.2f}" y="{inset["top"] + 16:.2f}" '
        f'font-family="Arial, sans-serif" font-size="11" font-weight="700">'
        "Detalle del primer minimo local</text>"
    )
    for tick in ticks(zoom_y_min, zoom_y_max, 3):
        y = inset_y(tick)
        elements.append(
            f'<line x1="{inset["left"]}" y1="{y:.2f}" '
            f'x2="{inset["left"] + inset["width"]:.2f}" y2="{y:.2f}" '
            f'stroke="#eeeeee" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{inset["left"] - 5:.2f}" y="{y + 3:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="8">{tick:.4f}</text>'
        )
    inset_points = " ".join(
        f"{inset_x(tau):.2f},{inset_y(value):.2f}"
        for tau, value in zoom_rows
    )
    elements.append(
        f'<polyline points="{inset_points}" fill="none" stroke="#5f6b7a" '
        f'stroke-width="1.8" stroke-linejoin="round"/>'
    )
    for tau, value in zoom_rows:
        color = "#8a2222" if tau == selected_tau else "#5f6b7a"
        radius = 3.8 if tau == selected_tau else 2.2
        elements.append(
            f'<circle cx="{inset_x(tau):.2f}" cy="{inset_y(value):.2f}" '
            f'r="{radius}" fill="{color}"/>'
        )
    for tick in [zoom_start, selected_tau, zoom_end]:
        elements.append(
            f'<text x="{inset_x(tick):.2f}" '
            f'y="{inset["top"] + inset["height"] + 13:.2f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="9">{tick}</text>'
        )
    elements.append(
        f'<text x="{inset["left"] + inset["width"]/2:.2f}" '
        f'y="{inset["top"] + inset["height"] + 28:.2f}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="9">'
        f'AMI({selected_tau - 1}) &gt; AMI({selected_tau}) &lt; AMI({selected_tau + 1})'
        "</text>"
    )

    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def write_fnn_svg(
    path: Path,
    rows: list[dict[str, Any]],
    selected_m: int,
    threshold: float,
) -> None:
    xs = [int(row["m"]) for row in rows]
    ys = [float(row["fnn_percent"]) for row in rows]
    write_line_chart_svg(
        path,
        xs,
        [{"label": "FNN %", "values": ys, "color": "#8a4f9f"}],
        "Falsos vecinos cercanos",
        "dimension m",
        "porcentaje FNN",
        vertical_lines=[(selected_m, f"m={selected_m}")],
        horizontal_lines=[(threshold, f"{threshold:g}%")],
        y_limits=(0.0, 100.0),
    )


def write_cao_svg(path: Path, rows: list[dict[str, Any]], selected_m: int) -> None:
    xs = [int(row["m"]) for row in rows]
    e1 = [float(row["E1"]) for row in rows]
    e2 = [float(row["E2"]) for row in rows]
    write_line_chart_svg(
        path,
        xs,
        [
            {"label": "E1", "values": e1, "color": "#5f6b7a"},
            {"label": "E2", "values": e2, "color": "#b45f06"},
        ],
        "Metodo de Cao",
        "dimension m",
        "valor",
        vertical_lines=[(selected_m, f"m elegida = {selected_m}")],
        y_limits=(0.0, 1.1),
    )


def write_line_chart_svg(
    path: Path,
    xs: list[int],
    series: list[dict[str, Any]],
    title: str,
    x_label: str,
    y_label: str,
    vertical_lines: list[tuple[int, str]] | None = None,
    horizontal_lines: list[tuple[float, str]] | None = None,
    y_limits: tuple[float, float] | None = None,
    width: int = 1040,
    height: int = 430,
) -> None:
    """Grafico SVG de lineas."""
    vertical_lines = vertical_lines or []
    horizontal_lines = horizontal_lines or []
    all_y = [value for item in series for value in item["values"] if math.isfinite(value)]
    x_min, x_max = min(xs), max(xs)
    if y_limits is None:
        y_min = min(all_y + [value for value, _ in horizontal_lines])
        y_max = max(all_y + [value for value, _ in horizontal_lines])
        y_min, y_max = expanded_range(y_min, y_max)
    else:
        y_min, y_max = y_limits
        if y_min >= y_max:
            raise ValueError("Los limites del eje Y deben cumplir y_min < y_max")

    margin = {"left": 78, "right": 38, "top": 58, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / max(1.0, x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

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
            f'font-family="Arial, sans-serif" font-size="11">{tick:.4g}</text>'
        )
    for value, label in horizontal_lines:
        y = y_coord(value)
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="#8a2222" stroke-width="1" stroke-dasharray="5,4"/>'
        )
        elements.append(
            f'<text x="{margin["left"] + plot_width - 4:.2f}" y="{y - 5:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#8a2222">{esc(label)}</text>'
        )
    for value, label in vertical_lines:
        x = x_coord(value)
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
        points = " ".join(
            f"{x_coord(x):.2f},{y_coord(y):.2f}"
            for x, y in zip(xs, item["values"])
            if math.isfinite(y)
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{item["color"]}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in zip(xs, item["values"]):
            if math.isfinite(y):
                elements.append(
                    f'<circle cx="{x_coord(x):.2f}" cy="{y_coord(y):.2f}" r="2.2" '
                    f'fill="{item["color"]}" opacity="0.85"/>'
                )

    for tick in x_ticks(x_min, x_max):
        x = x_coord(tick)
        elements.append(
            f'<text x="{x:.2f}" y="{height - 30}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11">{tick:g}</text>'
        )
    elements.append(
        f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(x_label)}</text>'
    )
    elements.append(
        f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(y_label)}</text>'
    )
    legend_x = width - 185
    for index, item in enumerate(series):
        elements.append(legend_item(legend_x, 28 + 18 * index, item["color"], item["label"]))
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def write_embedding_2d_svg(
    path: Path,
    train_vectors: list[list[float]],
    train_indices: list[int],
    test_vectors: list[list[float]],
    test_indices: list[int],
    title: str,
) -> None:
    train_selected = select_scatter_points(train_vectors, train_indices, GLOBAL_SCATTER_POINTS // 2)
    test_selected = select_scatter_points(test_vectors, test_indices, GLOBAL_SCATTER_POINTS // 2)
    points = [
        (vector[0], vector[1], "#5f6b7a", "train")
        for vector, _ in train_selected
    ] + [
        (vector[0], vector[1], "#b45f06", "test")
        for vector, _ in test_selected
    ]
    write_scatter_2d_svg(path, points, title, "z_t", "z_{t-tau}")


def write_single_embedding_2d_svg(
    path: Path,
    vectors: list[list[float]],
    indices: list[int],
    title: str,
    color: str,
) -> None:
    selected = select_scatter_points(vectors, indices, WINDOW_SCATTER_POINTS)
    points = [(vector[0], vector[1], color, "window") for vector, _ in selected]
    write_scatter_2d_svg(path, points, title, "z_t", "z_{t-tau}")


def write_scatter_2d_svg(
    path: Path,
    points: list[tuple[float, float, str, str]],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 720,
    height: int = 620,
) -> None:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = robust_range(x_values)
    y_min, y_max = robust_range(y_values)
    margin = {"left": 72, "right": 26, "top": 56, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = [
        svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="17" font-weight="700">{esc(title)}</text>'
        ),
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#222222" stroke-width="1"/>',
    ]
    for tick in ticks(x_min, x_max, 5):
        x = x_coord(tick)
        elements.append(
            f'<text x="{x:.2f}" y="{height - 31}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="10">{tick:.3g}</text>'
        )
    for tick in ticks(y_min, y_max, 5):
        y = y_coord(tick)
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="#eeeeee" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="10">{tick:.3g}</text>'
        )
    for x_value, y_value, color, _ in points:
        if x_min <= x_value <= x_max and y_min <= y_value <= y_max:
            elements.append(
                f'<circle cx="{x_coord(x_value):.2f}" cy="{y_coord(y_value):.2f}" '
                f'r="1.8" fill="{color}" opacity="0.38"/>'
            )
    elements.append(
        f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(x_label)}</text>'
    )
    elements.append(
        f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(y_label)}</text>'
    )
    if any(label == "train" for *_, label in points):
        elements.append(legend_item(width - 145, 38, "#5f6b7a", "train"))
        elements.append(legend_item(width - 145, 56, "#b45f06", "test"))
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def write_embedding_3d_svg(
    path: Path,
    train_vectors: list[list[float]],
    train_indices: list[int],
    test_vectors: list[list[float]],
    test_indices: list[int],
    title: str,
    width: int = 840,
    height: int = 650,
) -> None:
    """Proyeccion isometrica simple de las tres primeras coordenadas."""
    selected = [
        (vector[:3], "#5f6b7a")
        for vector, _ in select_scatter_points(train_vectors, train_indices, GLOBAL_SCATTER_POINTS // 2)
    ] + [
        (vector[:3], "#b45f06")
        for vector, _ in select_scatter_points(test_vectors, test_indices, GLOBAL_SCATTER_POINTS // 2)
    ]
    coords = [project_3d(*vector) for vector, _ in selected]
    x_min, x_max = robust_range([coord[0] for coord in coords])
    y_min, y_max = robust_range([coord[1] for coord in coords])
    margin = {"left": 52, "right": 32, "top": 58, "bottom": 42}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

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
    for (x_proj, y_proj), (_, color) in zip(coords, selected):
        if x_min <= x_proj <= x_max and y_min <= y_proj <= y_max:
            elements.append(
                f'<circle cx="{x_coord(x_proj):.2f}" cy="{y_coord(y_proj):.2f}" '
                f'r="1.8" fill="{color}" opacity="0.35"/>'
            )
    elements.append(legend_item(width - 155, 38, "#5f6b7a", "train"))
    elements.append(legend_item(width - 155, 56, "#b45f06", "test"))
    elements.append(
        f'<text x="{width/2:.1f}" y="{height - 12}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="12">'
        "Proyeccion visual de (z_t, z_{t-tau}, z_{t-2tau}); no es prueba de atractor</text>"
    )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def project_3d(x_value: float, y_value: float, z_value: float) -> tuple[float, float]:
    return (x_value - 0.55 * y_value, 0.5 * x_value + 0.5 * y_value - 0.9 * z_value)


def select_scatter_points(
    vectors: list[list[float]],
    indices: list[int],
    limit: int,
) -> list[tuple[list[float], int]]:
    if len(vectors) <= limit:
        return list(zip(vectors, indices))
    selected: list[tuple[list[float], int]] = []
    for output_index in range(limit):
        source_index = round((len(vectors) - 1) * output_index / (limit - 1))
        selected.append((vectors[source_index], indices[source_index]))
    return selected




def robust_range(values: list[float]) -> tuple[float, float]:
    sorted_values = sorted(values)
    lower = percentile(sorted_values, 0.01)
    upper = percentile(sorted_values, 0.99)
    if lower == upper:
        lower, upper = min(sorted_values), max(sorted_values)
    if lower == upper:
        return lower - 1.0, upper + 1.0
    padding = 0.06 * (upper - lower)
    return lower - padding, upper + padding


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


def expanded_range(y_min: float, y_max: float) -> tuple[float, float]:
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        return y_min - delta, y_max + delta
    padding = 0.07 * (y_max - y_min)
    return y_min - padding, y_max + padding


def ticks(y_min: float, y_max: float, count: int) -> list[float]:
    return [y_min + (y_max - y_min) * index / (count - 1) for index in range(count)]


def x_ticks(x_min: float, x_max: float) -> list[float]:
    if x_max <= 20:
        return list(range(int(x_min), int(x_max) + 1))
    step = max(1, round((x_max - x_min) / 6))
    return [x_min + step * index for index in range(int((x_max - x_min) // step) + 1)]




def rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}"
        else:
            rounded[key] = value
    return rounded


def legend_item(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<g><circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
        f'<text x="{x + 10:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
        f'font-size="12">{esc(label)}</text></g>'
    )


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 8: espacio de estados.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
