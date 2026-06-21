"""Fase 11: prediccion local en el espacio de estados."""

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
    continuous_block_positions,
    evaluate_predictions,
    knn_mean_predictions,
    load_ar_coefficients,
    metrics_to_row,
    nearest_neighbor_predictions,
    prepare_phase11_splits,
    read_prediction_series,
    sample_positions,
)


HORIZON = 12
K_GRID = [2, 3, 5, 10, 20, 50]
K_MAX = max(K_GRID)
EVAL_SAMPLE_SIZE = 5000
PLOT_BLOCK_SIZE = 720
RANDOM_SEED = 20260602


def main() -> int:
    args = build_parser().parse_args()
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    phase8_params = json.loads(args.phase8_params.read_text(encoding="utf-8"))
    tau = int(phase8_params["tau_selected"])
    dim = int(phase8_params["m_selected"])
    theiler_window = max(int(phase8_params.get("theiler_window", 0)), tau * dim)

    data = read_prediction_series(args.input)
    splits, z_values, x_mean_train, x_std_train, row_splits = prepare_phase11_splits(
        data,
        tau,
        dim,
    )
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    train_plus_validation = combine_splits("train_plus_validation", [train, validation])
    first_validation_index = row_splits["validation"][0]
    mean_y_train = mean_known_targets(train, first_validation_index, HORIZON)
    ar_coefficients = load_ar_coefficients(args.ar_coefficients)

    validation_positions = sample_positions(len(validation.vectors), EVAL_SAMPLE_SIZE)
    test_positions = sample_positions(len(test.vectors), EVAL_SAMPLE_SIZE)
    plot_positions = continuous_block_positions(len(test.vectors), PLOT_BLOCK_SIZE, center_fraction=0.45)

    print(
        "Fase 11: "
        f"tau={tau}, m={dim}, theiler={theiler_window}, "
        f"validation_eval={len(validation_positions)}, test_eval={len(test_positions)}"
    )
    print("Buscando vecinos para validation contra train...")
    validation_neighbors = compute_neighbor_sets(
        validation,
        train,
        validation_positions,
        K_MAX,
        theiler_window,
        HORIZON,
        progress_label="validation knn",
    )
    validation_rows = validation_k_selection_rows(
        validation,
        validation_positions,
        validation_neighbors,
        mean_y_train,
    )
    best_k = min(validation_rows, key=lambda row: float(row["rmse"]))["k"]
    best_k = int(best_k)
    print(f"k seleccionado por RMSE en validation: {best_k}")

    print("Buscando vecinos para test contra train+validation...")
    test_neighbors = compute_neighbor_sets(
        test,
        train_plus_validation,
        test_positions,
        K_MAX,
        theiler_window,
        HORIZON,
        progress_label="test knn",
    )

    test_metrics_rows, prediction_rows, test_predictions = evaluate_test_models(
        test,
        test_positions,
        test_neighbors,
        best_k,
        mean_y_train,
        z_values,
        ar_coefficients,
        x_mean_train,
        x_std_train,
    )

    print("Buscando vecinos para ventana continua de test...")
    plot_neighbors = compute_neighbor_sets(
        test,
        train_plus_validation,
        plot_positions,
        K_MAX,
        theiler_window,
        HORIZON,
        progress_label="test plot knn",
    )
    plot_rows = build_plot_rows(
        test,
        plot_positions,
        plot_neighbors,
        best_k,
        z_values,
        ar_coefficients,
        x_mean_train,
        x_std_train,
    )

    split_rows = build_split_summary_rows(
        data.times,
        splits,
        row_splits,
        validation_positions,
        test_positions,
    )
    align = alignment_check(data.x, data.y, HORIZON)
    summary = build_summary(
        tau,
        dim,
        theiler_window,
        mean_y_train,
        x_mean_train,
        x_std_train,
        best_k,
        align,
        validation_rows,
        test_metrics_rows,
        split_rows,
        phase8_params,
    )

    write_rows_csv(
        tables_dir / "phase11_split_summary.csv",
        split_rows,
        [
            "split",
            "row_start_time",
            "row_end_time",
            "row_n",
            "embedding_start_time",
            "embedding_end_time",
            "embedding_n",
            "evaluation_n",
        ],
    )
    write_rows_csv(
        tables_dir / "phase11_validation_k_selection.csv",
        validation_rows,
        ["k", "n", "mae", "mse", "rmse", "r2_oos", "bias_yhat_minus_y", "error_std"],
    )
    write_rows_csv(
        tables_dir / "phase11_test_metrics.csv",
        test_metrics_rows,
        ["model", "split", "n", "mae", "mse", "rmse", "r2_oos", "bias_yhat_minus_y", "error_std"],
    )
    write_rows_csv(
        tables_dir / "phase11_predictions_test_sample.csv",
        prediction_rows,
        [
            "open_time",
            "index",
            "y_true",
            "historical_mean",
            "persistence",
            "ar49",
            "nearest_neighbor",
            f"knn_mean_k{best_k}",
            "best_local_error",
        ],
    )
    (tables_dir / "phase11_prediction_summary.json").write_text(
        json.dumps(clean_json(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    write_validation_k_svg(figures_dir / "phase11_validation_k_selection.svg", validation_rows)
    write_real_vs_predicted_svg(figures_dir / "phase11_test_real_vs_predicted.svg", plot_rows, best_k)
    write_errors_time_svg(
        figures_dir / "phase11_test_errors_time.svg",
        [row["open_time"] for row in prediction_rows],
        test_predictions[f"knn_mean_k{best_k}_errors"],
        f"Errores en test del kNN local, k={best_k}",
    )
    write_error_histogram_svg(
        figures_dir / "phase11_test_error_histogram.svg",
        test_predictions[f"knn_mean_k{best_k}_errors"],
        f"Histograma de errores del kNN local, k={best_k}",
    )
    write_metrics_comparison_svg(
        figures_dir / "phase11_test_metrics_comparison.svg",
        test_metrics_rows,
    )
    print("Fase 11 completada.")
    return 0


def mean_known_targets(split: EmbeddedSplit, first_query_index: int, horizon: int) -> float:
    known = [
        target
        for target, index in zip(split.targets, split.indices)
        if index + horizon <= first_query_index
    ]
    if not known:
        raise ValueError("No hay targets historicos conocidos para la media")
    return sum(known) / len(known)


def validation_k_selection_rows(
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
        row["k"] = k
        rows.append(row)
    return rows


def evaluate_test_models(
    test: EmbeddedSplit,
    positions: list[int],
    neighbor_sets: list[Any],
    best_k: int,
    mean_y_train: float,
    z_values: list[float],
    ar_coefficients: list[float],
    x_mean_train: float,
    x_std_train: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    y_true = [test.targets[position] for position in positions]
    times = [test.times[position] for position in positions]
    indices = [test.indices[position] for position in positions]
    persistence = [test.persistence[position] for position in positions]
    historical_mean = [mean_y_train] * len(positions)
    ar49 = ar_recursive_forecast(
        z_values,
        indices,
        ar_coefficients,
        HORIZON,
        x_mean_train,
        x_std_train,
    )
    nearest = nearest_neighbor_predictions(neighbor_sets)
    knn_best = knn_mean_predictions(neighbor_sets, best_k)

    predictions_by_model = {
        "historical_mean": historical_mean,
        "persistence": persistence,
        "ar49_horizon12": ar49,
        "nearest_neighbor": nearest,
        f"knn_mean_k{best_k}": knn_best,
    }
    metric_rows = [
        metrics_to_row(evaluate_predictions(model, "test", y_true, pred, mean_y_train))
        for model, pred in predictions_by_model.items()
    ]
    prediction_rows: list[dict[str, Any]] = []
    for row_index, position in enumerate(positions):
        best_prediction = knn_best[row_index]
        prediction_rows.append(
            {
                "open_time": times[row_index],
                "index": indices[row_index],
                "y_true": y_true[row_index],
                "historical_mean": historical_mean[row_index],
                "persistence": persistence[row_index],
                "ar49": ar49[row_index],
                "nearest_neighbor": nearest[row_index],
                f"knn_mean_k{best_k}": best_prediction,
                "best_local_error": best_prediction - y_true[row_index],
            }
        )
    predictions_by_model[f"knn_mean_k{best_k}_errors"] = [
        pred - true for true, pred in zip(y_true, knn_best)
    ]
    return metric_rows, prediction_rows, predictions_by_model


def build_plot_rows(
    test: EmbeddedSplit,
    positions: list[int],
    neighbor_sets: list[Any],
    best_k: int,
    z_values: list[float],
    ar_coefficients: list[float],
    x_mean_train: float,
    x_std_train: float,
) -> list[dict[str, Any]]:
    indices = [test.indices[position] for position in positions]
    ar49 = ar_recursive_forecast(
        z_values,
        indices,
        ar_coefficients,
        HORIZON,
        x_mean_train,
        x_std_train,
    )
    knn_best = knn_mean_predictions(neighbor_sets, best_k)
    rows: list[dict[str, Any]] = []
    for output_index, position in enumerate(positions):
        rows.append(
            {
                "open_time": test.times[position],
                "index": test.indices[position],
                "y_true": test.targets[position],
                "persistence": test.persistence[position],
                "ar49": ar49[output_index],
                f"knn_mean_k{best_k}": knn_best[output_index],
            }
        )
    return rows


def build_split_summary_rows(
    all_times: list[str],
    splits: dict[str, EmbeddedSplit],
    row_splits: dict[str, list[int]],
    validation_positions: list[int],
    test_positions: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eval_counts = {"train": 0, "validation": len(validation_positions), "test": len(test_positions)}
    for name in ["train", "validation", "test"]:
        split = splits[name]
        raw_indices = row_splits[name]
        rows.append(
            {
                "split": name,
                "row_start_time": all_times[raw_indices[0]] if raw_indices else "",
                "row_end_time": all_times[raw_indices[-1]] if raw_indices else "",
                "row_n": len(raw_indices),
                "embedding_start_time": split.times[0] if split.times else "",
                "embedding_end_time": split.times[-1] if split.times else "",
                "embedding_n": len(split.vectors),
                "evaluation_n": eval_counts[name],
            }
        )
    return rows


def build_summary(
    tau: int,
    dim: int,
    theiler_window: int,
    mean_y_train: float,
    x_mean_train: float,
    x_std_train: float,
    best_k: int,
    align: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    test_metrics_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    phase8_params: dict[str, Any],
) -> dict[str, Any]:
    best_test = min(test_metrics_rows, key=lambda row: float(row["rmse"]))
    local_test = next(row for row in test_metrics_rows if row["model"] == f"knn_mean_k{best_k}")
    persistence_test = next(row for row in test_metrics_rows if row["model"] == "persistence")
    return {
        "series": "log_rv_past_12",
        "target": "log_rv_future_12",
        "tau": tau,
        "m": dim,
        "theiler_window": theiler_window,
        "horizon_bars": HORIZON,
        "horizon_minutes": HORIZON * 5,
        "k_grid": K_GRID,
        "selected_k_by_validation_rmse": best_k,
        "eval_sample_size_requested": EVAL_SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "x_mean_train": x_mean_train,
        "x_std_train": x_std_train,
        "mean_y_train_known": mean_y_train,
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
            "tau_selection_rule": phase8_params.get("tau_selection_rule"),
            "m_selection_rule": phase8_params.get("m_selection_rule"),
            "embedding_convention": phase8_params.get("embedding_convention"),
        },
        "validation_k_selection": validation_rows,
        "test_metrics": test_metrics_rows,
        "split_summary": split_rows,
        "best_test_model_by_rmse": best_test,
        "local_vs_persistence_rmse_delta": float(local_test["rmse"]) - float(persistence_test["rmse"]),
        "local_vs_persistence_mae_delta": float(local_test["mae"]) - float(persistence_test["mae"]),
    }




def write_validation_k_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    series = [
        ("MAE", [float(row["mae"]) for row in rows], "#2a6fbb"),
        ("RMSE", [float(row["rmse"]) for row in rows], "#b45f06"),
    ]
    ks = [int(row["k"]) for row in rows]
    write_xy_series_svg(path, ks, series, "Validation: seleccion de k", "k", "error")


def write_real_vs_predicted_svg(path: Path, rows: list[dict[str, Any]], best_k: int) -> None:
    times = [row["open_time"] for row in rows]
    series = [
        ("real", [float(row["y_true"]) for row in rows], "#222222"),
        ("persistencia", [float(row["persistence"]) for row in rows], "#6f7f8f"),
        ("AR(49)", [float(row["ar49"]) for row in rows], "#6b8e23"),
        (f"kNN k={best_k}", [float(row[f"knn_mean_k{best_k}"]) for row in rows], "#b45f06"),
    ]
    write_time_series_svg(path, times, series, f"Test: real vs predicho, kNN k={best_k}", "log RV futura")


def write_errors_time_svg(path: Path, times: list[str], errors: list[float], title: str) -> None:
    series = [("error", errors, "#8a2222")]
    write_time_series_svg(path, times, series, title, "y_hat - y")


def write_error_histogram_svg(path: Path, errors: list[float], title: str) -> None:
    width, height = 980, 430
    margin = {"left": 70, "right": 28, "top": 56, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    x_min, x_max = expanded_range(min(errors), max(errors), 0.06)
    bins = 36
    counts = [0] * bins
    for error in errors:
        index = int((error - x_min) / (x_max - x_min) * bins)
        counts[max(0, min(bins - 1, index))] += 1
    y_max = max(counts)

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * value / y_max

    elements = base_svg(width, height, margin, title, 0.0, y_max, y_coord, plot_width, plot_height)
    bin_width = plot_width / bins
    for index, count in enumerate(counts):
        x = margin["left"] + index * bin_width
        y = y_coord(count)
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bin_width*0.88:.2f}" '
            f'height="{margin["top"] + plot_height - y:.2f}" fill="#b45f06" opacity="0.72"/>'
        )
    if x_min < 0.0 < x_max:
        x_zero = x_coord(0.0)
        elements.append(
            f'<line x1="{x_zero:.2f}" y1="{margin["top"]}" x2="{x_zero:.2f}" '
            f'y2="{margin["top"] + plot_height:.2f}" stroke="#222222" stroke-width="1.6" '
            f'stroke-dasharray="4,3"/>'
        )
    elements.extend(x_axis_ticks(margin, plot_width, plot_height, x_min, x_max, x_coord, count=7))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "error = y_hat - y", "frecuencia"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_metrics_comparison_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1120, 500
    margin = {"left": 76, "right": 28, "top": 56, "bottom": 128}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    models = [row["model"] for row in rows]
    mae = [float(row["mae"]) for row in rows]
    rmse = [float(row["rmse"]) for row in rows]
    y_min, y_max = 0.0, max(mae + rmse) * 1.12

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, "Test: comparacion de MAE y RMSE", y_min, y_max, y_coord, plot_width, plot_height)
    slot_width = plot_width / len(models)
    for index, model in enumerate(models):
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
            f'font-size="11">{esc(model)}</text>'
        )
    elements.append(legend_item(width - 170, 28, "#2a6fbb", "MAE"))
    elements.append(legend_item(width - 170, 46, "#b45f06", "RMSE"))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "modelo", "error"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_xy_series_svg(
    path: Path,
    x_values: list[int],
    series: list[tuple[str, list[float], str]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 980, 440
    margin = {"left": 76, "right": 30, "top": 56, "bottom": 62}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    x_min, x_max = min(x_values), max(x_values)
    all_y = [value for _, values, _ in series for value in values]
    y_min, y_max = expanded_range(min(all_y), max(all_y), 0.08)

    def x_coord(value: float) -> float:
        if x_max == x_min:
            return margin["left"] + plot_width / 2
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    for label, values, color in series:
        points = " ".join(f"{x_coord(x):.2f},{y_coord(y):.2f}" for x, y in zip(x_values, values))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in zip(x_values, values):
            elements.append(f'<circle cx="{x_coord(x):.2f}" cy="{y_coord(y):.2f}" r="3.5" fill="{color}"/>')
    for x in x_values:
        elements.append(
            f'<text x="{x_coord(x):.2f}" y="{height - 36}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11">{x}</text>'
        )
    for index, (label, _, color) in enumerate(series):
        elements.append(legend_item(width - 145, 30 + 18 * index, color, label))
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
    width, height = 1180, 500
    margin = {"left": 76, "right": 32, "top": 56, "bottom": 66}
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
    tick_positions = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    for position in tick_positions:
        elements.append(
            f'<text x="{x_coord(position):.2f}" y="{height - 35}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="10">{esc(short_time(times[position]))}</text>'
        )
    for index, (label, _, color) in enumerate(series):
        elements.append(legend_line(width - 210, 28 + 18 * index, color, label))
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
        f'<text x="{width/2:.1f}" y="26" text-anchor="middle" font-family="Arial, sans-serif" '
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
        elements.append(
            f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" '
            f'y2="{bottom:.2f}" stroke="#f0f0f0"/>'
        )
        elements.append(f'<line x1="{x:.2f}" y1="{bottom:.2f}" x2="{x:.2f}" y2="{bottom + 5:.2f}" stroke="#222222"/>')
        elements.append(f'<text x="{x:.2f}" y="{bottom + 18:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{tick:.3g}</text>')
    return elements


def legend_item(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<g><circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
        f'<text x="{x + 10:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" '
        f'font-size="12">{esc(label)}</text></g>'
    )


def legend_line(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<g><line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 28:.2f}" y2="{y:.2f}" '
        f'stroke="{color}" stroke-width="2"/>'
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


def short_time(value: str) -> str:
    return value[:10] + "\n" + value[11:16]


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 11: prediccion local.")
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
