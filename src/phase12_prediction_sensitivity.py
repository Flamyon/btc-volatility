"""Fase 12: sensibilidad de la prediccion local."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from local_prediction import (
    EmbeddedSplit,
    alignment_check,
    ar_recursive_forecast,
    combine_splits,
    compute_neighbor_sets,
    evaluate_predictions,
    knn_mean_predictions,
    load_ar_coefficients,
    metrics_to_row,
    nearest_neighbor_predictions,
    prepare_phase11_splits,
    read_prediction_series,
    sample_positions,
    split_row_indices,
)


HORIZON = 12
TAU = 137
CONFIGS = [
    {"config_name": "tau137_m5", "tau": TAU, "m": 5, "theiler_window": TAU * 5, "eval_sample_size": 5000},
    {"config_name": "tau137_m14", "tau": TAU, "m": 14, "theiler_window": TAU * 14, "eval_sample_size": 1000},
]
K_GRID = [2, 3, 5, 10, 20, 50, 100, 200]
K_MAX = max(K_GRID)
EVAL_SAMPLE_SIZE = 5000
RANDOM_SEED = 20260602


def main() -> int:
    args = build_parser().parse_args()
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    phase8_params = json.loads(args.phase8_params.read_text(encoding="utf-8"))
    data = read_prediction_series(args.input)
    ar_coefficients = load_ar_coefficients(args.ar_coefficients)
    align = alignment_check(data.x, data.y, HORIZON)
    raw_splits = split_row_indices(data.times, HORIZON)
    first_validation_index = raw_splits["validation"][0]
    mean_y_train = mean_known_raw_targets(data.y, raw_splits["train"], first_validation_index, HORIZON)

    validation_rows: list[dict[str, Any]] = []
    test_metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    real_vs_pred_rows: list[dict[str, Any]] = []
    config_summaries: list[dict[str, Any]] = []

    print("Fase 12: sensibilidad de k y m")
    print(f"Alineacion max_abs_diff={align['max_abs_difference_y_t_vs_x_t_plus_horizon']:.3g}")
    for config in CONFIGS:
        result = run_config(
            config,
            data,
            ar_coefficients,
            mean_y_train,
        )
        validation_rows.extend(result["validation_rows"])
        test_metric_rows.extend(result["test_metric_rows"])
        comparison_rows.append(result["comparison_row"])
        config_summaries.append(result["summary"])
        if result["plot_rows"]:
            real_vs_pred_rows.extend(result["plot_rows"])

    summary = {
        "objective": "sensibilidad compacta de k y m para prediccion local",
        "series": "log_rv_past_12",
        "target": "log_rv_future_12",
        "horizon_bars": HORIZON,
        "horizon_minutes": HORIZON * 5,
        "k_grid": K_GRID,
        "eval_sample_size_default": EVAL_SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "alignment_check": align,
        "leakage_controls": {
            "purge_bars_between_splits": HORIZON,
            "validation_neighbors": "train only",
            "test_neighbors": "train+validation only",
            "candidate_label_rule": "candidate_index + horizon <= query_index",
            "theiler_rule": "abs(query_index - candidate_index) > theiler_window",
            "test_used_for_k_selection": False,
            "test_used_as_neighbor_library": False,
        },
        "phase8_reference": {
            "tau_selected": phase8_params.get("tau_selected"),
            "m_selected_fnn": phase8_params.get("m_selected"),
            "m_cao": phase8_params.get("m_cao"),
            "tau_selection_rule": phase8_params.get("tau_selection_rule"),
            "m_selection_rule": phase8_params.get("m_selection_rule"),
        },
        "configs": config_summaries,
        "config_comparison": comparison_rows,
        "test_metrics": test_metric_rows,
    }

    write_rows_csv(
        tables_dir / "phase12_validation_k_selection.csv",
        validation_rows,
        [
            "config_name",
            "tau",
            "m",
            "theiler_window",
            "k",
            "n",
            "mae",
            "mse",
            "rmse",
            "r2_oos",
            "bias_yhat_minus_y",
            "error_std",
        ],
    )
    write_rows_csv(
        tables_dir / "phase12_test_metrics.csv",
        test_metric_rows,
        [
            "config_name",
            "model",
            "split",
            "tau",
            "m",
            "theiler_window",
            "selected_k",
            "n",
            "mae",
            "mse",
            "rmse",
            "r2_oos",
            "bias_yhat_minus_y",
            "error_std",
        ],
    )
    write_rows_csv(
        tables_dir / "phase12_config_comparison.csv",
        comparison_rows,
        [
            "config_name",
            "tau",
            "m",
            "theiler_window",
            "selected_k",
            "validation_rmse_selected_k",
            "test_rmse_knn",
            "test_rmse_persistence",
            "test_rmse_ar49",
            "delta_rmse_knn_vs_persistence",
            "delta_rmse_knn_vs_ar49",
            "relative_improvement_vs_persistence_percent",
            "relative_difference_vs_ar49_percent",
        ],
    )
    (tables_dir / "phase12_prediction_summary.json").write_text(
        json.dumps(clean_json(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    write_validation_rmse_svg(figures_dir / "phase12_validation_rmse_by_k.svg", validation_rows)
    write_test_rmse_by_config_svg(figures_dir / "phase12_test_rmse_by_config.svg", comparison_rows)
    write_test_metrics_comparison_svg(figures_dir / "phase12_test_metrics_comparison.svg", test_metric_rows)
    if real_vs_pred_rows:
        write_real_vs_predicted_svg(
            figures_dir / "phase12_real_vs_predicted_best_configs.svg",
            real_vs_pred_rows,
        )

    print_final_summary(comparison_rows)
    return 0


def run_config(
    config: dict[str, Any],
    data: Any,
    ar_coefficients: list[float],
    mean_y_train: float,
) -> dict[str, Any]:
    config_name = str(config["config_name"])
    tau = int(config["tau"])
    dim = int(config["m"])
    theiler_window = int(config["theiler_window"])
    eval_sample_size = int(config.get("eval_sample_size", EVAL_SAMPLE_SIZE))
    splits, z_values, x_mean_train, x_std_train, row_splits = prepare_phase11_splits(data, tau, dim)
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    train_plus_validation = combine_splits("train_plus_validation", [train, validation])
    validation_positions = sample_positions(len(validation.vectors), eval_sample_size)
    test_positions = sample_positions(len(test.vectors), eval_sample_size)

    print(
        f"{config_name}: tau={tau}, m={dim}, theiler={theiler_window}, "
        f"validation_eval={len(validation_positions)}, test_eval={len(test_positions)}"
    )
    print(f"{config_name}: vecinos validation contra train...")
    validation_neighbors = compute_neighbor_sets(
        validation,
        train,
        validation_positions,
        K_MAX,
        theiler_window,
        HORIZON,
        progress_label=f"{config_name} validation",
        progress_every=progress_interval(len(validation_positions)),
    )
    validation_rows = validation_metrics_by_k(
        config,
        validation,
        validation_positions,
        validation_neighbors,
        mean_y_train,
    )
    selected = min(validation_rows, key=lambda row: float(row["rmse"]))
    selected_k = int(selected["k"])
    print(f"{config_name}: k seleccionado={selected_k}")

    print(f"{config_name}: vecinos test contra train+validation...")
    test_neighbors = compute_neighbor_sets(
        test,
        train_plus_validation,
        test_positions,
        K_MAX,
        theiler_window,
        HORIZON,
        progress_label=f"{config_name} test",
        progress_every=progress_interval(len(test_positions)),
    )
    test_metric_rows, plot_rows = test_metrics_for_config(
        config,
        test,
        test_positions,
        test_neighbors,
        selected_k,
        mean_y_train,
        z_values,
        ar_coefficients,
        x_mean_train,
        x_std_train,
    )
    comparison_row = config_comparison_row(config, selected, test_metric_rows)
    summary = {
        "config_name": config_name,
        "tau": tau,
        "m": dim,
        "theiler_window": theiler_window,
        "effective_history_bars": (dim - 1) * tau,
        "effective_history_minutes": (dim - 1) * tau * 5,
        "effective_history_hours": (dim - 1) * tau * 5 / 60.0,
        "train_embedding_n": len(train.vectors),
        "validation_embedding_n": len(validation.vectors),
        "test_embedding_n": len(test.vectors),
        "validation_eval_n": len(validation_positions),
        "test_eval_n": len(test_positions),
        "eval_sample_size_requested": eval_sample_size,
        "eval_sample_note": (
            "5000 puntos equiespaciados"
            if eval_sample_size == EVAL_SAMPLE_SIZE
            else "reducido por coste computacional de embedding alto"
        ),
        "selected_k": selected_k,
        "x_mean_train": x_mean_train,
        "x_std_train": x_std_train,
        "raw_split_train_n": len(row_splits["train"]),
        "raw_split_validation_n": len(row_splits["validation"]),
        "raw_split_test_n": len(row_splits["test"]),
    }
    return {
        "validation_rows": validation_rows,
        "test_metric_rows": test_metric_rows,
        "comparison_row": comparison_row,
        "summary": summary,
        "plot_rows": plot_rows,
    }


def validation_metrics_by_k(
    config: dict[str, Any],
    validation: EmbeddedSplit,
    positions: list[int],
    neighbor_sets: list[Any],
    mean_y_train: float,
) -> list[dict[str, Any]]:
    y_true = [validation.targets[position] for position in positions]
    rows: list[dict[str, Any]] = []
    for k in K_GRID:
        predictions = knn_mean_predictions(neighbor_sets, k)
        metrics = evaluate_predictions(f"knn_mean_k{k}", "validation", y_true, predictions, mean_y_train)
        row = metrics_to_row(metrics)
        rows.append(add_config_fields(config, row | {"k": k}))
    return rows


def test_metrics_for_config(
    config: dict[str, Any],
    test: EmbeddedSplit,
    positions: list[int],
    neighbor_sets: list[Any],
    selected_k: int,
    mean_y_train: float,
    z_values: list[float],
    ar_coefficients: list[float],
    x_mean_train: float,
    x_std_train: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_true = [test.targets[position] for position in positions]
    query_indices = [test.indices[position] for position in positions]
    historical_mean = [mean_y_train] * len(positions)
    persistence = [test.persistence[position] for position in positions]
    ar49 = ar_recursive_forecast(
        z_values,
        query_indices,
        ar_coefficients,
        HORIZON,
        x_mean_train,
        x_std_train,
    )
    nearest = nearest_neighbor_predictions(neighbor_sets)
    knn = knn_mean_predictions(neighbor_sets, selected_k)
    predictions = {
        "historical_mean": historical_mean,
        "persistence": persistence,
        "ar49_horizon12": ar49,
        "nearest_neighbor": nearest,
        f"knn_mean_k{selected_k}": knn,
    }
    rows: list[dict[str, Any]] = []
    for model, values in predictions.items():
        metric_row = metrics_to_row(evaluate_predictions(model, "test", y_true, values, mean_y_train))
        metric_row = add_config_fields(config, metric_row)
        metric_row["selected_k"] = selected_k
        rows.append(metric_row)

    plot_rows: list[dict[str, Any]] = []
    for position_index in sample_positions(len(positions), min(720, len(positions))):
        source_position = positions[position_index]
        plot_rows.append(
            {
                "config_name": config["config_name"],
                "open_time": test.times[source_position],
                "index": test.indices[source_position],
                "y_true": y_true[position_index],
                f"knn_mean_k{selected_k}": knn[position_index],
                "selected_k": selected_k,
            }
        )
    return rows, plot_rows


def add_config_fields(config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_name": config["config_name"],
        "tau": config["tau"],
        "m": config["m"],
        "theiler_window": config["theiler_window"],
        **row,
    }


def config_comparison_row(
    config: dict[str, Any],
    selected_validation_row: dict[str, Any],
    test_metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_k = int(selected_validation_row["k"])
    knn = next(row for row in test_metric_rows if row["model"] == f"knn_mean_k{selected_k}")
    persistence = next(row for row in test_metric_rows if row["model"] == "persistence")
    ar49 = next(row for row in test_metric_rows if row["model"] == "ar49_horizon12")
    rmse_knn = float(knn["rmse"])
    rmse_persistence = float(persistence["rmse"])
    rmse_ar49 = float(ar49["rmse"])
    return {
        "config_name": config["config_name"],
        "tau": config["tau"],
        "m": config["m"],
        "theiler_window": config["theiler_window"],
        "selected_k": selected_k,
        "validation_rmse_selected_k": float(selected_validation_row["rmse"]),
        "test_rmse_knn": rmse_knn,
        "test_rmse_persistence": rmse_persistence,
        "test_rmse_ar49": rmse_ar49,
        "delta_rmse_knn_vs_persistence": rmse_knn - rmse_persistence,
        "delta_rmse_knn_vs_ar49": rmse_knn - rmse_ar49,
        "relative_improvement_vs_persistence_percent": 100.0 * (rmse_persistence - rmse_knn) / rmse_persistence,
        "relative_difference_vs_ar49_percent": 100.0 * (rmse_knn - rmse_ar49) / rmse_ar49,
    }


def mean_known_raw_targets(
    targets: list[float],
    train_indices: list[int],
    first_query_index: int,
    horizon: int,
) -> float:
    known = [
        targets[index]
        for index in train_indices
        if index + horizon <= first_query_index
    ]
    if not known:
        raise ValueError("No hay targets historicos conocidos")
    return sum(known) / len(known)


def progress_interval(n: int) -> int:
    return 250 if n <= 1000 else 1000




def write_validation_rmse_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    configs = sorted({str(row["config_name"]) for row in rows})
    k_values = K_GRID
    series = []
    colors = {"tau137_m5": "#2a6fbb", "tau137_m14": "#b45f06"}
    for config in configs:
        values = [
            float(next(row for row in rows if row["config_name"] == config and int(row["k"]) == k)["rmse"])
            for k in k_values
        ]
        series.append((config, values, colors.get(config, "#555555")))
    write_xy_series_svg(path, k_values, series, "Validation: RMSE por k", "k", "RMSE")


def write_test_rmse_by_config_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 920, 460
    margin = {"left": 76, "right": 34, "top": 58, "bottom": 92}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    configs = [row["config_name"] for row in rows]
    knn_values = [float(row["test_rmse_knn"]) for row in rows]
    persistence = float(rows[0]["test_rmse_persistence"])
    ar49 = float(rows[0]["test_rmse_ar49"])
    y_min, y_max = 0.0, max(knn_values + [persistence, ar49]) * 1.12

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, "Test: RMSE kNN por configuracion", y_min, y_max, y_coord, plot_width, plot_height)
    slot_width = plot_width / len(configs)
    for index, (config, value) in enumerate(zip(configs, knn_values)):
        center = margin["left"] + slot_width * (index + 0.5)
        bar_width = min(120.0, slot_width * 0.44)
        y = y_coord(value)
        elements.append(
            f'<rect x="{center - bar_width/2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{margin["top"] + plot_height - y:.2f}" fill="#2a6fbb" opacity="0.78"/>'
        )
        elements.append(
            f'<text x="{center:.2f}" y="{height - 48}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="12">{esc(config)}</text>'
        )
    for value, color, label, y_offset in [
        (persistence, "#6f7f8f", "persistencia", 0),
        (ar49, "#6b8e23", "AR(49)", 18),
    ]:
        y = y_coord(value)
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" '
            f'y2="{y:.2f}" stroke="{color}" stroke-width="2.2" stroke-dasharray="6,4"/>'
        )
        elements.append(legend_line(width - 190, 28 + y_offset, color, label))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "configuracion", "RMSE"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_test_metrics_comparison_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1120, 500
    margin = {"left": 76, "right": 28, "top": 58, "bottom": 128}
    baseline_config = "tau137_m5"
    baseline_rows = [row for row in rows if row["config_name"] == baseline_config]
    models: list[tuple[str, dict[str, Any]]] = []
    for model in ["historical_mean", "persistence", "ar49_horizon12"]:
        models.append((model, next(row for row in baseline_rows if row["model"] == model)))
    for config in ["tau137_m5", "tau137_m14"]:
        knn_row = next(row for row in rows if row["config_name"] == config and str(row["model"]).startswith("knn_mean_k"))
        models.append((f"knn {config}", knn_row))

    labels = [label for label, _ in models]
    mae = [float(row["mae"]) for _, row in models]
    rmse = [float(row["rmse"]) for _, row in models]
    y_min, y_max = 0.0, max(mae + rmse) * 1.12
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, "Test: MAE y RMSE", y_min, y_max, y_coord, plot_width, plot_height)
    slot_width = plot_width / len(labels)
    for index, label in enumerate(labels):
        center = margin["left"] + slot_width * (index + 0.5)
        for offset, value, color in [(-0.18, mae[index], "#2a6fbb"), (0.18, rmse[index], "#b45f06")]:
            bar_width = slot_width * 0.26
            x = center + offset * slot_width - bar_width / 2
            y = y_coord(value)
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{margin["top"] + plot_height - y:.2f}" fill="{color}" opacity="0.78"/>'
            )
        label_y = height - 78
        elements.append(
            f'<text x="{center:.2f}" y="{label_y}" text-anchor="end" '
            f'transform="rotate(-32 {center:.2f},{label_y})" font-family="Arial, sans-serif" '
            f'font-size="11">{esc(label)}</text>'
        )
    elements.append(legend_item(width - 160, 30, "#2a6fbb", "MAE"))
    elements.append(legend_item(width - 160, 48, "#b45f06", "RMSE"))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "modelo", "error"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_real_vs_predicted_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    # Figura opcional compacta: se muestran predicciones kNN de ambas configuraciones.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["config_name"]), []).append(row)
    configs = sorted(grouped)
    width, height = 1180, 520
    margin = {"left": 76, "right": 32, "top": 58, "bottom": 68}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    base_rows = grouped[configs[0]]
    times = [row["open_time"] for row in base_rows]
    series: list[tuple[str, list[float], str]] = [
        ("real", [float(row["y_true"]) for row in base_rows], "#222222")
    ]
    colors = {"tau137_m5": "#2a6fbb", "tau137_m14": "#b45f06"}
    for config in configs:
        k = int(grouped[config][0]["selected_k"])
        series.append((f"{config} k={k}", [float(row[f"knn_mean_k{k}"]) for row in grouped[config]], colors.get(config, "#555555")))
    write_time_series_svg(path, times, series, "Test: real vs kNN por configuracion", "log RV futura")


def write_xy_series_svg(
    path: Path,
    x_values: list[int],
    series: list[tuple[str, list[float], str]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 980, 440
    margin = {"left": 76, "right": 30, "top": 58, "bottom": 64}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    x_min, x_max = min(x_values), max(x_values)
    all_y = [value for _, values, _ in series for value in values]
    y_min, y_max = expanded_range(min(all_y), max(all_y), 0.08)

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    for label, values, color in series:
        points = " ".join(f"{x_coord(k):.2f},{y_coord(value):.2f}" for k, value in zip(x_values, values))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for k, value in zip(x_values, values):
            elements.append(f'<circle cx="{x_coord(k):.2f}" cy="{y_coord(value):.2f}" r="3.5" fill="{color}"/>')
    for k in x_values:
        elements.append(
            f'<text x="{x_coord(k):.2f}" y="{height - 36}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11">{k}</text>'
        )
    for index, (label, _, color) in enumerate(series):
        elements.append(legend_item(width - 160, 30 + 18 * index, color, label))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, x_label, y_label))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_time_series_svg(
    path: Path,
    times: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    y_label: str,
) -> None:
    width, height = 1180, 520
    margin = {"left": 76, "right": 32, "top": 58, "bottom": 68}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    all_y = [value for _, values, _ in series for value in values]
    y_min, y_max = expanded_range(min(all_y), max(all_y), 0.06)
    n = len(times)

    def x_coord(index: int) -> float:
        return margin["left"] + plot_width * index / max(1, n - 1)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    for label, values, color in series:
        points = " ".join(f"{x_coord(index):.2f},{y_coord(value):.2f}" for index, value in enumerate(values))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    for position in sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1])):
        elements.append(
            f'<text x="{x_coord(position):.2f}" y="{height - 35}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="10">{esc(times[position][:16])}</text>'
        )
    for index, (label, _, color) in enumerate(series):
        elements.append(legend_line(width - 220, 30 + 18 * index, color, label))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "tiempo", y_label))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def base_svg(
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
        f'stroke="{color}" stroke-width="2.2" stroke-dasharray="6,4"/>'
        f'<text x="{x + 36:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
        f'font-size="12">{esc(label)}</text></g>'
    )




def rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}" if math.isfinite(value) else "nan"
        else:
            rounded[key] = value
    return rounded


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def expanded_range(y_min: float, y_max: float, pad_fraction: float) -> tuple[float, float]:
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        return y_min - delta, y_max + delta
    padding = pad_fraction * (y_max - y_min)
    return y_min - padding, y_max + padding


def ticks(y_min: float, y_max: float, count: int) -> list[float]:
    return [y_min + (y_max - y_min) * index / (count - 1) for index in range(count)]


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def print_final_summary(rows: list[dict[str, Any]]) -> None:
    m5 = next(row for row in rows if row["config_name"] == "tau137_m5")
    m14 = next(row for row in rows if row["config_name"] == "tau137_m14")
    persistence = float(m5["test_rmse_persistence"])
    ar49 = float(m5["test_rmse_ar49"])
    print("Resumen Fase 12")
    print(f"mejor k m=5: {m5['selected_k']}")
    print(f"mejor k m=14: {m14['selected_k']}")
    print(f"RMSE test kNN m=5: {float(m5['test_rmse_knn']):.6g}")
    print(f"RMSE test kNN m=14: {float(m14['test_rmse_knn']):.6g}")
    print(f"RMSE test persistencia: {persistence:.6g}")
    print(f"RMSE test AR(49): {ar49:.6g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 12: sensibilidad predictiva.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument(
        "--phase8-params",
        type=Path,
        default=Path("reports/tables/phase8_selected_embedding_params.json"),
    )
    parser.add_argument(
        "--ar-coefficients",
        type=Path,
        default=Path("reports/tables/phase6_ar_coefficients.csv"),
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
