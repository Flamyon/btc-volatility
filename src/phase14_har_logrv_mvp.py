"""Fase 14: HAR-logRV compacto y exportable para MVP."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from har_logrv import (
    HAR_FEATURES,
    HAR_TARGET,
    HARLogRVModel,
    fit_har_logrv_ols,
    har_feature_row,
    predict_har_logrv,
    save_har_artifact,
)
from local_prediction import (
    ar_recursive_forecast,
    combine_splits,
    compute_neighbor_sets,
    continuous_block_positions,
    evaluate_predictions,
    knn_mean_predictions,
    load_ar_coefficients,
    metrics_to_row,
    prepare_phase11_splits,
    read_prediction_series,
    sample_positions,
    split_row_indices,
)


HORIZON = 12
TAU = 137
DIM = 5
KNN_K = 200
THEILER_WINDOW = TAU * DIM
TEST_SAMPLE_SIZE = 5000
PLOT_BLOCK_SIZE = 720

CORE_MODEL_ORDER = ["ar49_horizon12", "knn_tau137_m5_k200", "har_logrv_compact"]
MODEL_DISPLAY_NAMES = {
    "historical_mean": "Media historica",
    "persistence": "Persistencia",
    "ar49_horizon12": "AR(49)",
    "knn_tau137_m5_k200": "kNN k=200",
    "har_logrv_compact": "HAR-logRV",
}


def main() -> int:
    args = build_parser().parse_args()
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    model_artifacts_dir = args.model_artifacts_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows = read_har_rows(args.input)
    data = read_prediction_series(args.input)
    ar_coefficients = load_ar_coefficients(args.ar_coefficients)
    raw_splits = split_indices(rows)
    first_validation_index = raw_splits["validation"][0]
    mean_y_train = mean_known_targets(rows, raw_splits["train"], first_validation_index)
    align = alignment_check(rows)

    print("Fase 14: HAR-logRV compacto")
    print(f"Filas validas={len(rows)}, train={len(raw_splits['train'])}, validation={len(raw_splits['validation'])}, test={len(raw_splits['test'])}")
    print(f"Alineacion max_abs_diff={align['max_abs_difference_y_t_vs_x_t_plus_horizon']:.3g}")

    har_model = fit_har_logrv_ols(
        [har_feature_row(rows[index]) for index in raw_splits["train"]],
        [float(rows[index][HAR_TARGET]) for index in raw_splits["train"]],
    )

    splits, z_values, x_mean_train, x_std_train, row_splits = prepare_phase11_splits(data, TAU, DIM)
    train_plus_validation = combine_splits("train_plus_validation", [splits["train"], splits["validation"]])
    test = splits["test"]
    validation_raw_indices = raw_splits["validation"]
    test_raw_indices = raw_splits["test"]

    validation_rows = evaluate_full_split(
        "validation_full",
        rows,
        validation_raw_indices,
        har_model,
        mean_y_train,
        z_values,
        ar_coefficients,
        x_mean_train,
        x_std_train,
    )

    print("Calculando kNN tau=137, m=5, k=200 en muestra test comparable...")
    test_positions = sample_positions(len(test.vectors), TEST_SAMPLE_SIZE)
    test_neighbors = compute_neighbor_sets(
        test,
        train_plus_validation,
        test_positions,
        KNN_K,
        THEILER_WINDOW,
        HORIZON,
        progress_label="phase14 test knn",
        progress_every=1000,
    )
    comparable_predictions = comparable_sample_predictions(
        rows,
        test,
        test_positions,
        test_neighbors,
        har_model,
        mean_y_train,
        z_values,
        ar_coefficients,
        x_mean_train,
        x_std_train,
    )
    test_metric_rows = build_test_metric_rows(comparable_predictions, mean_y_train)
    comparable_prediction_rows = comparable_prediction_output_rows(comparable_predictions)

    full_test_rows = evaluate_full_split(
        "test_full",
        rows,
        test_raw_indices,
        har_model,
        mean_y_train,
        z_values,
        ar_coefficients,
        x_mean_train,
        x_std_train,
    )
    test_metric_rows.extend(full_test_rows)

    comparison_rows = model_comparison_rows(test_metric_rows)
    prediction_sample_rows = prediction_sample(comparable_predictions, sample_size=500)
    coefficient_rows = coefficient_table(har_model)
    mvp_rows, mvp_model = run_mvp_1000_mode(rows)

    train_metrics = metrics_to_row(
        evaluate_predictions(
            "har_logrv_compact",
            "train",
            [float(rows[index][HAR_TARGET]) for index in raw_splits["train"]],
            predict_har_logrv(har_model, [rows[index] for index in raw_splits["train"]]),
            mean_y_train,
        )
    )
    har_validation = next(row for row in validation_rows if row["model"] == "har_logrv_compact")
    har_test = next(row for row in test_metric_rows if row["model"] == "har_logrv_compact" and row["split"] == "test_knn_comparable_sample")

    artifact_metadata = {
        "trained_at": "2026-06-06",
        "train_start": rows[raw_splits["train"][0]]["open_time"],
        "train_end": rows[raw_splits["train"][-1]]["open_time"],
        "validation_start": rows[raw_splits["validation"][0]]["open_time"],
        "validation_end": rows[raw_splits["validation"][-1]]["open_time"],
        "test_start": rows[raw_splits["test"][0]]["open_time"],
        "test_end": rows[raw_splits["test"][-1]]["open_time"],
        "horizon_bars": HORIZON,
        "horizon_minutes": HORIZON * 5,
    }
    artifact_metrics = {
        "train_rmse": train_metrics["rmse"],
        "validation_rmse": har_validation["rmse"],
        "test_rmse": har_test["rmse"],
        "test_split_for_artifact_metric": "test_knn_comparable_sample",
    }
    save_har_artifact(tables_dir / "phase14_har_model_artifact.json", har_model, artifact_metrics, artifact_metadata)
    save_har_artifact(model_artifacts_dir / "har_logrv_model.json", har_model, artifact_metrics, artifact_metadata)

    summary = {
        "objective": "HAR-logRV compacto exportable para MVP",
        "target": HAR_TARGET,
        "features": HAR_FEATURES,
        "horizon_bars": HORIZON,
        "horizon_minutes": HORIZON * 5,
        "split": {
            "train": [rows[raw_splits["train"][0]]["open_time"], rows[raw_splits["train"][-1]]["open_time"], len(raw_splits["train"])],
            "validation": [rows[raw_splits["validation"][0]]["open_time"], rows[raw_splits["validation"][-1]]["open_time"], len(raw_splits["validation"])],
            "test": [rows[raw_splits["test"][0]]["open_time"], rows[raw_splits["test"][-1]]["open_time"], len(raw_splits["test"])],
        },
        "alignment_check": align,
        "leakage_controls": {
            "purge_bars_between_splits": HORIZON,
            "har_train_only": True,
            "features_are_past_only": HAR_FEATURES,
            "target": HAR_TARGET,
            "test_used_for_training": False,
            "test_used_for_feature_selection": False,
            "knn_k_fixed_from_phase12": KNN_K,
            "knn_neighbors_for_test": "train+validation only",
        },
        "har_coefficients": coefficient_rows,
        "test_metrics": test_metric_rows,
        "mvp_1000_mode": mvp_rows,
    }
    (tables_dir / "phase14_prediction_summary.json").write_text(
        json.dumps(clean_json(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    write_rows_csv(
        tables_dir / "phase14_har_coefficients.csv",
        coefficient_rows,
        ["term", "feature", "coefficient"],
    )
    write_rows_csv(
        tables_dir / "phase14_validation_metrics.csv",
        validation_rows,
        metric_columns(),
    )
    write_rows_csv(
        tables_dir / "phase14_test_metrics.csv",
        test_metric_rows,
        metric_columns(),
    )
    write_rows_csv(
        tables_dir / "phase14_predictions_test_sample.csv",
        prediction_sample_rows,
        [
            "open_time",
            "index",
            "y_true",
            "historical_mean",
            "persistence",
            "ar49_horizon12",
            "knn_tau137_m5_k200",
            "har_logrv_compact",
            "har_error",
        ],
    )
    write_rows_csv(
        tables_dir / "phase14_predictions_test_comparable.csv",
        comparable_prediction_rows,
        [
            "open_time",
            "index",
            "y_true",
            "ar49_horizon12",
            "knn_tau137_m5_k200",
            "har_logrv_compact",
            "ar49_error",
            "knn_error",
            "har_error",
            "ar49_abs_error",
            "knn_abs_error",
            "har_abs_error",
        ],
    )
    write_rows_csv(
        tables_dir / "phase14_model_comparison.csv",
        comparison_rows,
        [
            "model",
            "split",
            "n",
            "mae",
            "rmse",
            "r2_oos",
            "delta_rmse_vs_har",
            "relative_rmse_vs_har_percent",
            "rank_rmse",
        ],
    )
    write_rows_csv(
        tables_dir / "phase14_mvp_1000_mode_summary.csv",
        mvp_rows,
        [
            "n_input_rows",
            "n_effective_rows",
            "train_n",
            "test_n",
            "rmse_test_mvp",
            "mae_test_mvp",
            "r2_oos_test_mvp",
            "last_timestamp",
            "last_log_rv_past_12",
            "last_log_rv_past_48",
            "last_log_rv_past_288",
            "predicted_log_rv_future_12",
            "intercept",
            "beta_log_rv_past_12",
            "beta_log_rv_past_48",
            "beta_log_rv_past_288",
        ],
    )

    plot_rows = build_plot_rows(
        rows,
        test,
        train_plus_validation,
        har_model,
        z_values,
        ar_coefficients,
        x_mean_train,
        x_std_train,
    )
    write_metrics_comparison_svg(figures_dir / "phase14_test_metrics_comparison.svg", comparison_rows)
    write_core_model_metrics_svg(figures_dir / "phase14_ar_knn_har_metrics.svg", comparison_rows)
    write_real_vs_predicted_svg(figures_dir / "phase14_real_vs_predicted.svg", plot_rows)
    write_ar_knn_har_real_vs_predicted_svg(
        figures_dir / "phase14_ar_knn_har_real_vs_predicted.svg",
        plot_rows,
        comparison_rows,
    )
    write_core_model_error_distribution_svg(
        figures_dir / "phase14_ar_knn_har_error_distribution.svg",
        comparable_prediction_rows,
        comparison_rows,
    )
    write_errors_time_svg(figures_dir / "phase14_errors_time.svg", rows, test_raw_indices, har_model)
    write_error_histogram_svg(figures_dir / "phase14_error_histogram.svg", rows, test_raw_indices, har_model)
    write_coefficients_svg(figures_dir / "phase14_har_coefficients.svg", coefficient_rows)
    write_mvp_context_svg(figures_dir / "phase14_mvp_1000_prediction_context.svg", rows[-1000:], mvp_rows[0])

    print_final_summary(comparison_rows, mvp_rows)
    return 0


def read_har_rows(path: Path) -> list[dict[str, Any]]:
    required = {"open_time", *HAR_FEATURES, HAR_TARGET}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
        for row in reader:
            parsed: dict[str, Any] = {"open_time": row["open_time"]}
            valid = True
            for column in [*HAR_FEATURES, HAR_TARGET]:
                value = float(row[column])
                if not math.isfinite(value):
                    valid = False
                    break
                parsed[column] = value
            if valid:
                rows.append(parsed)
    if not rows:
        raise ValueError("No hay filas validas para HAR-logRV")
    return rows


def split_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    splits = split_row_indices([str(row["open_time"]) for row in rows], HORIZON)
    for name, indices in splits.items():
        if not indices:
            raise ValueError(f"Split vacio: {name}")
    return splits


def mean_known_targets(rows: list[dict[str, Any]], train_indices: list[int], first_query_index: int) -> float:
    known = [
        float(rows[index][HAR_TARGET])
        for index in train_indices
        if index + HORIZON <= first_query_index
    ]
    if not known:
        raise ValueError("No hay targets historicos conocidos para la media")
    return sum(known) / len(known)


def alignment_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diffs = [
        abs(float(rows[index][HAR_TARGET]) - float(rows[index + HORIZON]["log_rv_past_12"]))
        for index in range(len(rows) - HORIZON)
    ]
    return {
        "horizon": HORIZON,
        "n_checked": len(diffs),
        "max_abs_difference_y_t_vs_x_t_plus_horizon": max(diffs),
        "mean_abs_difference_y_t_vs_x_t_plus_horizon": sum(diffs) / len(diffs),
    }


def evaluate_full_split(
    split_name: str,
    rows: list[dict[str, Any]],
    indices: list[int],
    har_model: HARLogRVModel,
    mean_y_train: float,
    z_values: list[float],
    ar_coefficients: list[float],
    x_mean_train: float,
    x_std_train: float,
) -> list[dict[str, Any]]:
    selected_rows = [rows[index] for index in indices]
    y_true = [float(row[HAR_TARGET]) for row in selected_rows]
    predictions = {
        "historical_mean": [mean_y_train] * len(indices),
        "persistence": [float(row["log_rv_past_12"]) for row in selected_rows],
        "ar49_horizon12": ar_recursive_forecast(z_values, indices, ar_coefficients, HORIZON, x_mean_train, x_std_train),
        "har_logrv_compact": predict_har_logrv(har_model, selected_rows),
    }
    return [
        metrics_to_row(evaluate_predictions(model, split_name, y_true, pred, mean_y_train))
        for model, pred in predictions.items()
    ]


def comparable_sample_predictions(
    rows: list[dict[str, Any]],
    test: Any,
    test_positions: list[int],
    test_neighbors: list[Any],
    har_model: HARLogRVModel,
    mean_y_train: float,
    z_values: list[float],
    ar_coefficients: list[float],
    x_mean_train: float,
    x_std_train: float,
) -> dict[str, Any]:
    query_indices = [test.indices[position] for position in test_positions]
    row_subset = [rows[index] for index in query_indices]
    y_true = [test.targets[position] for position in test_positions]
    return {
        "split": "test_knn_comparable_sample",
        "open_time": [test.times[position] for position in test_positions],
        "index": query_indices,
        "y_true": y_true,
        "historical_mean": [mean_y_train] * len(test_positions),
        "persistence": [test.persistence[position] for position in test_positions],
        "ar49_horizon12": ar_recursive_forecast(z_values, query_indices, ar_coefficients, HORIZON, x_mean_train, x_std_train),
        "knn_tau137_m5_k200": knn_mean_predictions(test_neighbors, KNN_K),
        "har_logrv_compact": predict_har_logrv(har_model, row_subset),
    }


def build_test_metric_rows(predictions: dict[str, Any], mean_y_train: float) -> list[dict[str, Any]]:
    y_true = predictions["y_true"]
    rows: list[dict[str, Any]] = []
    for model in ["historical_mean", "persistence", "ar49_horizon12", "knn_tau137_m5_k200", "har_logrv_compact"]:
        rows.append(metrics_to_row(evaluate_predictions(model, predictions["split"], y_true, predictions[model], mean_y_train)))
    return rows


def model_comparison_rows(test_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in test_metric_rows if row["split"] == "test_knn_comparable_sample"]
    har_rmse = float(next(row for row in rows if row["model"] == "har_logrv_compact")["rmse"])
    ordered = sorted(rows, key=lambda row: float(row["rmse"]))
    ranks = {row["model"]: rank for rank, row in enumerate(ordered, start=1)}
    return [
        {
            "model": row["model"],
            "split": row["split"],
            "n": row["n"],
            "mae": row["mae"],
            "rmse": row["rmse"],
            "r2_oos": row["r2_oos"],
            "delta_rmse_vs_har": float(row["rmse"]) - har_rmse,
            "relative_rmse_vs_har_percent": 100.0 * (float(row["rmse"]) - har_rmse) / har_rmse,
            "rank_rmse": ranks[row["model"]],
        }
        for row in rows
    ]


def prediction_sample(predictions: dict[str, Any], sample_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in sample_positions(len(predictions["y_true"]), sample_size):
        har_pred = float(predictions["har_logrv_compact"][position])
        y_true = float(predictions["y_true"][position])
        rows.append(
            {
                "open_time": predictions["open_time"][position],
                "index": predictions["index"][position],
                "y_true": y_true,
                "historical_mean": predictions["historical_mean"][position],
                "persistence": predictions["persistence"][position],
                "ar49_horizon12": predictions["ar49_horizon12"][position],
                "knn_tau137_m5_k200": predictions["knn_tau137_m5_k200"][position],
                "har_logrv_compact": har_pred,
                "har_error": har_pred - y_true,
            }
        )
    return rows


def comparable_prediction_output_rows(predictions: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, y_true_value in enumerate(predictions["y_true"]):
        y_true = float(y_true_value)
        ar_pred = float(predictions["ar49_horizon12"][position])
        knn_pred = float(predictions["knn_tau137_m5_k200"][position])
        har_pred = float(predictions["har_logrv_compact"][position])
        ar_error = ar_pred - y_true
        knn_error = knn_pred - y_true
        har_error = har_pred - y_true
        rows.append(
            {
                "open_time": predictions["open_time"][position],
                "index": predictions["index"][position],
                "y_true": y_true,
                "ar49_horizon12": ar_pred,
                "knn_tau137_m5_k200": knn_pred,
                "har_logrv_compact": har_pred,
                "ar49_error": ar_error,
                "knn_error": knn_error,
                "har_error": har_error,
                "ar49_abs_error": abs(ar_error),
                "knn_abs_error": abs(knn_error),
                "har_abs_error": abs(har_error),
            }
        )
    return rows


def coefficient_table(model: HARLogRVModel) -> list[dict[str, Any]]:
    rows = [{"term": "intercept", "feature": "intercept", "coefficient": model.intercept}]
    rows.extend(
        {"term": f"beta_{index}", "feature": feature, "coefficient": coefficient}
        for index, (feature, coefficient) in enumerate(zip(model.feature_names, model.coefficients), start=1)
    )
    return rows


def run_mvp_1000_mode(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], HARLogRVModel]:
    input_rows = rows[-1000:]
    effective_rows = [
        row for row in input_rows
        if all(math.isfinite(float(row[column])) for column in [*HAR_FEATURES, HAR_TARGET])
    ]
    split = int(0.70 * len(effective_rows))
    train_rows = effective_rows[: split - HORIZON]
    test_rows = effective_rows[split:]
    model = fit_har_logrv_ols(
        [har_feature_row(row) for row in train_rows],
        [float(row[HAR_TARGET]) for row in train_rows],
    )
    mean_train = sum(float(row[HAR_TARGET]) for row in train_rows) / len(train_rows)
    y_true = [float(row[HAR_TARGET]) for row in test_rows]
    y_pred = predict_har_logrv(model, test_rows)
    metrics = metrics_to_row(evaluate_predictions("har_logrv_compact", "mvp_1000_test", y_true, y_pred, mean_train))
    last = effective_rows[-1]
    last_prediction = predict_har_logrv(model, [last])[0]
    row = {
        "n_input_rows": len(input_rows),
        "n_effective_rows": len(effective_rows),
        "train_n": len(train_rows),
        "test_n": len(test_rows),
        "rmse_test_mvp": metrics["rmse"],
        "mae_test_mvp": metrics["mae"],
        "r2_oos_test_mvp": metrics["r2_oos"],
        "last_timestamp": last["open_time"],
        "last_log_rv_past_12": last["log_rv_past_12"],
        "last_log_rv_past_48": last["log_rv_past_48"],
        "last_log_rv_past_288": last["log_rv_past_288"],
        "predicted_log_rv_future_12": last_prediction,
        "intercept": model.intercept,
        "beta_log_rv_past_12": model.coefficients[0],
        "beta_log_rv_past_48": model.coefficients[1],
        "beta_log_rv_past_288": model.coefficients[2],
    }
    return [row], model


def build_plot_rows(
    rows: list[dict[str, Any]],
    test: Any,
    train_plus_validation: Any,
    har_model: HARLogRVModel,
    z_values: list[float],
    ar_coefficients: list[float],
    x_mean_train: float,
    x_std_train: float,
) -> list[dict[str, Any]]:
    positions = continuous_block_positions(len(test.vectors), min(PLOT_BLOCK_SIZE, len(test.vectors)), center_fraction=0.45)
    neighbors = compute_neighbor_sets(
        test,
        train_plus_validation,
        positions,
        KNN_K,
        THEILER_WINDOW,
        HORIZON,
        progress_label="phase14 plot knn",
        progress_every=250,
    )
    query_indices = [test.indices[position] for position in positions]
    har = predict_har_logrv(har_model, [rows[index] for index in query_indices])
    ar49 = ar_recursive_forecast(z_values, query_indices, ar_coefficients, HORIZON, x_mean_train, x_std_train)
    knn = knn_mean_predictions(neighbors, KNN_K)
    return [
        {
            "open_time": test.times[position],
            "y_true": test.targets[position],
            "ar49_horizon12": ar49[offset],
            "knn_tau137_m5_k200": knn[offset],
            "har_logrv_compact": har[offset],
        }
        for offset, position in enumerate(positions)
    ]


def metric_columns() -> list[str]:
    return ["model", "split", "n", "mae", "mse", "rmse", "r2_oos", "bias_yhat_minus_y", "error_std"]


def write_metrics_comparison_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    write_metric_bars_svg(
        path,
        rows,
        "Fase 14: MAE y RMSE en test comparable",
        label_map={},
        subtitle=None,
    )


def write_core_model_metrics_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    write_metric_bars_svg(
        path,
        core_model_rows(rows),
        "Comparacion historica AR(49), kNN y HAR-logRV",
        label_map=MODEL_DISPLAY_NAMES,
        subtitle="Target: log_rv_future_12 · test comparable: 5000 puntos",
    )


def core_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model = {row["model"]: row for row in rows}
    missing = [model for model in CORE_MODEL_ORDER if model not in by_model]
    if missing:
        raise ValueError(f"Faltan modelos para comparacion central: {missing}")
    return [by_model[model] for model in CORE_MODEL_ORDER]


def core_metric_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["model"]: row for row in core_model_rows(rows)}


def write_metric_bars_svg(
    path: Path,
    rows: list[dict[str, Any]],
    title: str,
    label_map: dict[str, str],
    subtitle: str | None,
) -> None:
    labels = [label_map.get(row["model"], row["model"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    rmse = [float(row["rmse"]) for row in rows]
    width, height = 1120, 500
    margin = {"left": 76, "right": 28, "top": 78 if subtitle else 58, "bottom": 130}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    y_min, y_max = 0.0, max(mae + rmse) * 1.12

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    if subtitle:
        elements.append(f'<text x="{width/2:.1f}" y="50" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#444444">{esc(subtitle)}</text>')
    slot_width = plot_width / len(labels)
    for index, label in enumerate(labels):
        center = margin["left"] + slot_width * (index + 0.5)
        for offset, value, color in [(-0.18, mae[index], "#2a6fbb"), (0.18, rmse[index], "#b45f06")]:
            bar_width = slot_width * 0.24
            x = center + offset * slot_width - bar_width / 2
            y = y_coord(value)
            elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{margin["top"] + plot_height - y:.2f}" fill="{color}" opacity="0.78"/>')
            elements.append(f'<text x="{x + bar_width/2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{value:.6f}</text>')
        label_y = height - 78
        elements.append(f'<text x="{center:.2f}" y="{label_y}" text-anchor="end" transform="rotate(-32 {center:.2f},{label_y})" font-family="Arial, sans-serif" font-size="11">{esc(label)}</text>')
    elements.append(legend_item(width - 150, 30, "#2a6fbb", "MAE"))
    elements.append(legend_item(width - 150, 48, "#b45f06", "RMSE"))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "modelo", "error"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_real_vs_predicted_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    times = [row["open_time"] for row in rows]
    series = [
        ("real", [float(row["y_true"]) for row in rows], "#222222"),
        ("AR(49)", [float(row["ar49_horizon12"]) for row in rows], "#6b8e23"),
        ("kNN k=200", [float(row["knn_tau137_m5_k200"]) for row in rows], "#b45f06"),
        ("HAR-logRV", [float(row["har_logrv_compact"]) for row in rows], "#2a6fbb"),
    ]
    write_time_series_svg(path, times, series, "Fase 14: real vs predicho", "log_rv_future_12")


def write_ar_knn_har_real_vs_predicted_svg(
    path: Path,
    rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
) -> None:
    metrics = core_metric_lookup(metrics_rows)
    times = [row["open_time"] for row in rows]
    series = [
        ("real", [float(row["y_true"]) for row in rows], "#222222"),
        (f"AR(49), RMSE={float(metrics['ar49_horizon12']['rmse']):.6f}", [float(row["ar49_horizon12"]) for row in rows], "#6b8e23"),
        (f"kNN k=200, RMSE={float(metrics['knn_tau137_m5_k200']['rmse']):.6f}", [float(row["knn_tau137_m5_k200"]) for row in rows], "#b45f06"),
        (f"HAR-logRV, RMSE={float(metrics['har_logrv_compact']['rmse']):.6f}", [float(row["har_logrv_compact"]) for row in rows], "#2a6fbb"),
    ]
    write_time_series_svg(
        path,
        times,
        series,
        "Real vs predicho: AR(49), kNN y HAR-logRV",
        "log_rv_future_12",
        subtitle="Ventana representativa del test historico",
    )


def write_core_model_error_distribution_svg(
    path: Path,
    prediction_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
) -> None:
    metrics = core_metric_lookup(metrics_rows)
    model_specs = [
        ("AR(49)", "ar49_abs_error", "ar49_horizon12", "#6b8e23"),
        ("kNN k=200", "knn_abs_error", "knn_tau137_m5_k200", "#b45f06"),
        ("HAR-logRV", "har_abs_error", "har_logrv_compact", "#2a6fbb"),
    ]
    distributions = []
    for label, column, model_key, color in model_specs:
        values = sorted(float(row[column]) for row in prediction_rows)
        distributions.append(
            {
                "label": label,
                "color": color,
                "rmse": float(metrics[model_key]["rmse"]),
                "p05": percentile(values, 0.05),
                "q1": percentile(values, 0.25),
                "median": percentile(values, 0.50),
                "q3": percentile(values, 0.75),
                "p95": percentile(values, 0.95),
            }
        )

    width, height = 980, 520
    margin = {"left": 82, "right": 32, "top": 84, "bottom": 116}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    y_min = 0.0
    y_max = max(item["p95"] for item in distributions) * 1.18

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, "Distribucion del error absoluto por modelo", y_min, y_max, y_coord, plot_width, plot_height)
    elements.append(f'<text x="{width/2:.1f}" y="50" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#444444">AR(49), kNN k=200 y HAR-logRV sobre test comparable</text>')
    slot_width = plot_width / len(distributions)
    for index, item in enumerate(distributions):
        center = margin["left"] + slot_width * (index + 0.5)
        box_width = slot_width * 0.36
        q1_y = y_coord(item["q1"])
        q3_y = y_coord(item["q3"])
        median_y = y_coord(item["median"])
        p05_y = y_coord(item["p05"])
        p95_y = y_coord(item["p95"])
        elements.append(f'<line x1="{center:.2f}" y1="{p95_y:.2f}" x2="{center:.2f}" y2="{p05_y:.2f}" stroke="{item["color"]}" stroke-width="1.8"/>')
        elements.append(f'<line x1="{center - box_width*0.28:.2f}" y1="{p95_y:.2f}" x2="{center + box_width*0.28:.2f}" y2="{p95_y:.2f}" stroke="{item["color"]}" stroke-width="1.8"/>')
        elements.append(f'<line x1="{center - box_width*0.28:.2f}" y1="{p05_y:.2f}" x2="{center + box_width*0.28:.2f}" y2="{p05_y:.2f}" stroke="{item["color"]}" stroke-width="1.8"/>')
        elements.append(f'<rect x="{center - box_width/2:.2f}" y="{q3_y:.2f}" width="{box_width:.2f}" height="{q1_y - q3_y:.2f}" fill="{item["color"]}" opacity="0.34" stroke="{item["color"]}" stroke-width="1.8"/>')
        elements.append(f'<line x1="{center - box_width/2:.2f}" y1="{median_y:.2f}" x2="{center + box_width/2:.2f}" y2="{median_y:.2f}" stroke="#222222" stroke-width="2"/>')
        elements.append(f'<text x="{center:.2f}" y="{height - 76}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{esc(item["label"])}</text>')
        elements.append(f'<text x="{center:.2f}" y="{height - 58}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">RMSE={item["rmse"]:.6f}</text>')
        elements.append(f'<text x="{center:.2f}" y="{height - 42}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">mediana={item["median"]:.4f}</text>')
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "modelo", "abs(error)"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_errors_time_svg(path: Path, rows: list[dict[str, Any]], indices: list[int], model: HARLogRVModel) -> None:
    positions = sample_positions(len(indices), min(1200, len(indices)))
    selected = [rows[indices[position]] for position in positions]
    predictions = predict_har_logrv(model, selected)
    errors = [pred - float(row[HAR_TARGET]) for pred, row in zip(predictions, selected)]
    series = [("error HAR", errors, "#2a6fbb")]
    write_time_series_svg(path, [row["open_time"] for row in selected], series, "Fase 14: errores HAR en test", "error")


def write_error_histogram_svg(path: Path, rows: list[dict[str, Any]], indices: list[int], model: HARLogRVModel) -> None:
    selected = [rows[index] for index in indices]
    predictions = predict_har_logrv(model, selected)
    errors = [pred - float(row[HAR_TARGET]) for pred, row in zip(predictions, selected)]
    write_histogram_svg(path, errors, "Fase 14: histograma errores HAR", "error HAR = y_hat - y")


def write_coefficients_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    beta_rows = [row for row in rows if row["term"] != "intercept"]
    labels = [row["feature"] for row in beta_rows]
    values = [float(row["coefficient"]) for row in beta_rows]
    write_bar_svg(path, labels, values, "Fase 14: coeficientes HAR-logRV", "coeficiente")


def write_mvp_context_svg(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    recent = rows[-300:]
    times = [row["open_time"] for row in recent]
    values = [float(row["log_rv_past_12"]) for row in recent]
    prediction = float(summary["predicted_log_rv_future_12"])
    width, height = 1180, 520
    margin = {"left": 82, "right": 32, "top": 58, "bottom": 72}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    all_y = values + [prediction]
    y_min, y_max = expanded_range(min(all_y), max(all_y), 0.08)
    n = len(values) + 1

    def x_coord(index: int) -> float:
        return margin["left"] + plot_width * index / max(1, n - 1)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, "MVP 1000 velas: contexto y prediccion HAR", y_min, y_max, y_coord, plot_width, plot_height)
    points = " ".join(f"{x_coord(index):.2f},{y_coord(value):.2f}" for index, value in enumerate(values))
    elements.append(f'<polyline points="{points}" fill="none" stroke="#2a6fbb" stroke-width="1.6"/>')
    pred_x = x_coord(n - 1)
    pred_y = y_coord(prediction)
    elements.append(f'<line x1="{x_coord(len(values)-1):.2f}" y1="{pred_y:.2f}" x2="{pred_x:.2f}" y2="{pred_y:.2f}" stroke="#b45f06" stroke-width="2" stroke-dasharray="5,4"/>')
    elements.append(f'<circle cx="{pred_x:.2f}" cy="{pred_y:.2f}" r="5.5" fill="#b45f06"/>')
    for position in sorted(set([0, len(values) // 2, len(values) - 1])):
        elements.append(f'<text x="{x_coord(position):.2f}" y="{height - 38}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{esc(times[position][:16])}</text>')
    elements.append(f'<text x="{pred_x:.2f}" y="{height - 38}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">pred +1h</text>')
    elements.append(legend_line(width - 250, 30, "#2a6fbb", "log_rv_past_12"))
    elements.append(legend_line(width - 250, 48, "#b45f06", f"HAR pred={prediction:.4g}"))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "tiempo", "logRV"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_bar_svg(path: Path, labels: list[str], values: list[float], title: str, y_label: str) -> None:
    width, height = 760, 460
    margin = {"left": 84, "right": 28, "top": 58, "bottom": 100}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    y_min, y_max = expanded_range(min(0.0, min(values)), max(0.0, max(values)), 0.12)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    zero_y = y_coord(0.0)
    elements.append(f'<line x1="{margin["left"]}" y1="{zero_y:.2f}" x2="{margin["left"] + plot_width:.2f}" y2="{zero_y:.2f}" stroke="#333333" stroke-width="1.2"/>')
    slot = plot_width / len(labels)
    for index, (label, value) in enumerate(zip(labels, values)):
        center = margin["left"] + slot * (index + 0.5)
        width_bar = slot * 0.42
        y = y_coord(value)
        top = min(y, zero_y)
        height_bar = abs(zero_y - y)
        color = "#2a6fbb" if value >= 0 else "#b45f06"
        elements.append(f'<rect x="{center - width_bar/2:.2f}" y="{top:.2f}" width="{width_bar:.2f}" height="{height_bar:.2f}" fill="{color}" opacity="0.78"/>')
        elements.append(f'<text x="{center:.2f}" y="{height - 54}" text-anchor="end" transform="rotate(-26 {center:.2f},{height - 54})" font-family="Arial, sans-serif" font-size="11">{esc(label)}</text>')
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "feature", y_label))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_histogram_svg(path: Path, values: list[float], title: str, x_label: str) -> None:
    bins = 40
    x_min, x_max = expanded_range(min(values), max(values), 0.04)
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int((value - x_min) / (x_max - x_min) * bins)))
        counts[index] += 1
    width, height = 880, 460
    margin = {"left": 76, "right": 28, "top": 58, "bottom": 64}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    y_min, y_max = 0.0, max(counts) * 1.12

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    bar_width = plot_width / bins
    for index, count in enumerate(counts):
        x = margin["left"] + index * bar_width
        y = y_coord(count)
        elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width*0.94:.2f}" height="{margin["top"] + plot_height - y:.2f}" fill="#2a6fbb" opacity="0.76"/>')
    elements.extend(x_axis_ticks(margin, plot_width, plot_height, x_min, x_max, x_coord, count=7))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, x_label, "frecuencia"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_time_series_svg(
    path: Path,
    times: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    y_label: str,
    subtitle: str | None = None,
) -> None:
    width, height = 1180, 520
    margin = {"left": 82, "right": 32, "top": 78 if subtitle else 58, "bottom": 72}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    all_y = [value for _, values, _ in series for value in values]
    y_min, y_max = expanded_range(min(all_y), max(all_y), 0.06)
    n = len(series[0][1])

    def x_coord(index: int) -> float:
        return margin["left"] + plot_width * index / max(1, n - 1)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    if subtitle:
        elements.append(f'<text x="{width/2:.1f}" y="50" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#444444">{esc(subtitle)}</text>')
    for label, values, color in series:
        points = " ".join(f"{x_coord(index):.2f},{y_coord(value):.2f}" for index, value in enumerate(values))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    for position in sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1])):
        label = times[min(position, len(times) - 1)][:16]
        elements.append(f'<text x="{x_coord(position):.2f}" y="{height - 38}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{esc(label)}</text>')
    for index, (label, _, color) in enumerate(series):
        elements.append(legend_line(width - 350, 30 + 18 * index, color, label))
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
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700">{esc(title)}</text>',
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width:.2f}" height="{plot_height:.2f}" fill="none" stroke="#222222"/>',
    ]
    for tick in ticks(y_min, y_max, 5):
        y = y_coord(tick)
        elements.append(f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" y2="{y:.2f}" stroke="#e8e8e8"/>')
        elements.append(f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11">{tick:.3g}</text>')
    return elements


def axis_labels(width: int, height: int, margin: dict[str, int], plot_width: float, plot_height: float, x_label: str, y_label: str) -> str:
    return "\n".join([
        f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(x_label)}</text>',
        f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(y_label)}</text>',
    ])


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


def legend_item(x: float, y: float, color: str, label: str) -> str:
    return f'<g><circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/><text x="{x + 10:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" font-size="12">{esc(label)}</text></g>'


def legend_line(x: float, y: float, color: str, label: str) -> str:
    return f'<g><line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 28:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"/><text x="{x + 36:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" font-size="12">{esc(label)}</text></g>'




def rounded_row(row: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = f"{value:.6g}" if math.isfinite(value) else "nan"
        else:
            rounded[key] = value
    return rounded


def expanded_range(y_min: float, y_max: float, pad_fraction: float) -> tuple[float, float]:
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        return y_min - delta, y_max + delta
    padding = pad_fraction * (y_max - y_min)
    return y_min - padding, y_max + padding


def ticks(y_min: float, y_max: float, count: int) -> list[float]:
    return [y_min + (y_max - y_min) * index / (count - 1) for index in range(count)]


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("No se puede calcular percentil sobre lista vacia")
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def print_final_summary(comparison_rows: list[dict[str, Any]], mvp_rows: list[dict[str, Any]]) -> None:
    print("Resumen Fase 14")
    for row in sorted(comparison_rows, key=lambda item: int(item["rank_rmse"])):
        print(f"{row['rank_rmse']}. {row['model']}: RMSE={float(row['rmse']):.6g}, MAE={float(row['mae']):.6g}")
    mvp = mvp_rows[0]
    print(f"MVP 1000: n={mvp['n_effective_rows']}, RMSE={float(mvp['rmse_test_mvp']):.6g}, pred_final={float(mvp['predicted_log_rv_future_12']):.6g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 14: HAR-logRV compacto.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_5m_features.csv"))
    parser.add_argument("--ar-coefficients", type=Path, default=Path("reports/tables/phase6_ar_coefficients.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-artifacts-dir", type=Path, default=Path("data/model_artifacts"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
