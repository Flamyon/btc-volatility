"""Fase 13: validacion metodologica con mapa logistico."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

from data_loading import write_rows_csv
from dynamics_quantification import (
    contiguous_block,
    rosenstein_curve,
    summarize_lyapunov,
)
from linear_filtering import ARModel, select_ar_yule_walker
from local_prediction import (
    EmbeddedSplit,
    ar_recursive_forecast,
    combine_splits,
    compute_neighbor_sets,
    continuous_block_positions,
    evaluate_predictions,
    knn_mean_predictions,
    metrics_to_row,
    nearest_neighbor_predictions,
    sample_positions,
)
from state_space import (
    average_mutual_information_from_bins,
    build_embedding_rows,
    discretize,
    fnn_and_cao,
    quantile_edges,
    select_m_from_cao,
    select_m_from_fnn,
    standardize_train,
)
from synthetic_logistic import SyntheticSeries, build_logistic_series


R = 4.0
X0 = 0.123456789
N_TOTAL = 12000
BURN_IN = 1000
SEED = 20260603
AMI_MAX_LAG = 50
AMI_BINS = 32
MAX_DIM = 10
FNN_SAMPLE_SIZE = 1200
FNN_THRESHOLD_PERCENT = 1.0
K_GRID = [1, 2, 3, 5, 10, 20, 50]
K_MAX = max(K_GRID)
HORIZON = 1
MAX_AR_ORDER = 100
LYAP_BLOCK_SIZE = 1800
LYAP_K_MAX = 20
LYAP_FIT_START = 1
LYAP_FIT_END = 6
PREDICTION_SHORT_WINDOW_SIZE = 50
PREDICTION_SHORT_WINDOW_CENTER_FRACTIONS = (0.20, 0.50, 0.80)


def main() -> int:
    args = build_parser().parse_args()
    synthetic_dir = args.synthetic_dir
    reports_dir = args.reports_dir
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    series_list = build_logistic_series(R, X0, N_TOTAL, BURN_IN, SEED)
    for series in series_list:
        write_series_csv(synthetic_dir / f"{series.name}.csv", series)

    write_series_examples_svg(figures_dir / "phase13_logistic_series_examples.svg", series_list)

    ami_rows: list[dict[str, Any]] = []
    fnn_rows: list[dict[str, Any]] = []
    cao_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    lyapunov_rows: list[dict[str, Any]] = []
    lyapunov_summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    prediction_metric_rows: list[dict[str, Any]] = []
    prediction_summary: list[dict[str, Any]] = []
    ar_order_rows: list[dict[str, Any]] = []

    embeddings_for_figures: dict[str, dict[str, Any]] = {}

    print("Fase 13: mapa logistico")
    for series in series_list:
        print(f"Procesando {series.name}...")
        train_end, validation_end = split_points(len(series.values))
        z_values, train_mean, train_std = standardize_train(series.values, train_end)
        current_ami = compute_ami(series.name, z_values, train_end)
        ami_rows.extend(current_ami)
        tau, tau_note = select_tau_operational(current_ami)
        theiler_for_fnn = max(1, tau)
        current_fnn, current_cao, _ = fnn_and_cao(
            z_values,
            train_end,
            tau,
            MAX_DIM,
            FNN_SAMPLE_SIZE,
            theiler_for_fnn,
            rtol=10.0,
            atol=2.0,
        )
        for row in current_fnn:
            fnn_rows.append({"series_name": series.name, "tau": tau, **row})
        for row in current_cao:
            cao_rows.append({"series_name": series.name, "tau": tau, **row})
        m_fnn, m_fnn_note = select_m_from_fnn(current_fnn, FNN_THRESHOLD_PERCENT)
        m_cao, m_cao_note = select_m_from_cao(current_cao, tolerance=0.03)
        m_selected, selection_note = select_embedding_dimension(m_fnn, m_cao, m_fnn_note, m_cao_note)
        selected_rows.append(
            {
                "series_name": series.name,
                "tau_selected": tau,
                "m_fnn": m_fnn,
                "m_cao": m_cao,
                "m_selected": m_selected,
                "tau_selection_note": tau_note,
                "selection_note": selection_note,
            }
        )

        embeddings_for_figures[series.name] = {
            "z_values": z_values,
            "tau": tau,
            "m": m_selected,
            "train_end": train_end,
            "validation_end": validation_end,
            "train_mean": train_mean,
            "train_std": train_std,
        }
        lyap_curve, lyap_summary = run_lyapunov(series.name, z_values, tau, m_selected)
        lyapunov_rows.extend(lyap_curve)
        lyapunov_summary_rows.append(lyap_summary)

        val_rows, pred_rows, pred_summary, current_ar_rows = run_prediction(
            series.name,
            series.values,
            z_values,
            train_mean,
            train_std,
            tau,
            m_selected,
            train_end,
            validation_end,
        )
        validation_rows.extend(val_rows)
        prediction_metric_rows.extend(pred_rows)
        prediction_summary.append(pred_summary)
        ar_order_rows.extend(current_ar_rows)

    write_rows_csv(tables_dir / "phase13_ami.csv", ami_rows, ["series_name", "tau", "ami"])
    write_rows_csv(
        tables_dir / "phase13_fnn.csv",
        fnn_rows,
        ["series_name", "tau", "m", "fnn_fraction", "fnn_percent", "n_used", "false_neighbors", "theiler_window", "rtol", "atol"],
    )
    write_rows_csv(
        tables_dir / "phase13_cao.csv",
        cao_rows,
        ["series_name", "tau", "m", "E1", "E2", "E_m", "E_star_m", "n_used"],
    )
    write_rows_csv(
        tables_dir / "phase13_selected_embedding_params.csv",
        selected_rows,
        ["series_name", "tau_selected", "m_fnn", "m_cao", "m_selected", "tau_selection_note", "selection_note"],
    )
    write_rows_csv(
        tables_dir / "phase13_lyapunov_rosenstein.csv",
        lyapunov_rows,
        ["series_name", "tau", "m", "k", "mean_log_distance", "n_pairs", "theiler_window"],
    )
    write_rows_csv(
        tables_dir / "phase13_lyapunov_summary.csv",
        lyapunov_summary_rows,
        ["series_name", "tau", "m", "slope_per_step", "fit_start", "fit_end", "r2_fit", "theoretical_ln2", "n_fit_points"],
    )
    write_rows_csv(
        tables_dir / "phase13_validation_k_selection.csv",
        validation_rows,
        ["series_name", "tau", "m", "theiler_window", "k", "n", "mae", "rmse", "r2_oos"],
    )
    write_rows_csv(
        tables_dir / "phase13_prediction_metrics.csv",
        prediction_metric_rows,
        ["series_name", "model", "split", "tau", "m", "theiler_window", "selected_k", "ar_order", "n", "mae", "rmse", "r2_oos"],
    )
    write_rows_csv(
        tables_dir / "phase13_ar_order_selection.csv",
        ar_order_rows,
        ["series_name", "p", "nobs_train", "innovation_variance", "aic", "bic"],
    )
    summary = {
        "system": "logistic map",
        "equation": "x[t+1] = r*x[t]*(1-x[t])",
        "r": R,
        "x0": X0,
        "n_total": N_TOTAL,
        "burn_in": BURN_IN,
        "seed": SEED,
        "output_length_after_burn_in": len(series_list[0].values),
        "noise": [
            {
                "series_name": series.name,
                "noise_level": series.noise_level,
                "noise_sigma": series.noise_sigma,
                "clipped_count": series.clipped_count,
            }
            for series in series_list
        ],
        "selected_embedding": selected_rows,
        "lyapunov": lyapunov_summary_rows,
        "prediction": prediction_summary,
        "ar_order_selection": selected_ar_order_rows(ar_order_rows),
        "theoretical_lyapunov_ln2": math.log(2.0),
    }
    (tables_dir / "phase13_prediction_summary.json").write_text(
        json.dumps(clean_json(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    write_ami_svg(figures_dir / "phase13_ami.svg", ami_rows, selected_rows)
    write_fnn_svg(figures_dir / "phase13_fnn.svg", fnn_rows)
    write_cao_svg(figures_dir / "phase13_cao.svg", cao_rows)
    write_embedding_figures(figures_dir, embeddings_for_figures)
    write_lyapunov_svg(figures_dir / "phase13_lyapunov_rosenstein.svg", lyapunov_rows)
    write_prediction_k_svg(figures_dir / "phase13_prediction_validation_k.svg", validation_rows)
    write_ar_bic_selection_svg(figures_dir / "phase13_ar_bic_selection.svg", ar_order_rows)
    clean_series = next(series for series in series_list if series.name == "logistic_clean")
    write_transition_map_clean_svg(
        figures_dir / "phase13_transition_map_clean.svg",
        clean_series.values,
        split_points(len(clean_series.values))[0],
    )
    short_prediction_windows = prediction_short_windows(len(clean_series.values))
    for series in series_list:
        write_prediction_real_vs_predicted_svg(
            figures_dir / f"phase13_prediction_real_vs_predicted_{suffix(series.name)}.svg",
            series.name,
            prediction_metric_rows,
            embeddings_for_figures[series.name],
            series.values,
        )
        write_prediction_real_vs_predicted_svg(
            figures_dir / f"phase13_prediction_real_vs_predicted_{suffix(series.name)}_compact.svg",
            series.name,
            prediction_metric_rows,
            embeddings_for_figures[series.name],
            series.values,
            compact=True,
        )
        for window_number, time_window in enumerate(short_prediction_windows, start=1):
            write_prediction_real_vs_predicted_svg(
                figures_dir / f"phase13_prediction_real_vs_predicted_{suffix(series.name)}_short_{window_number}.svg",
                series.name,
                prediction_metric_rows,
                embeddings_for_figures[series.name],
                series.values,
                time_window=time_window,
            )
    write_prediction_metrics_svg(figures_dir / "phase13_prediction_metrics.svg", prediction_metric_rows)
    print_final_summary(selected_rows, lyapunov_summary_rows, prediction_summary)
    return 0


def write_series_csv(path: Path, series: SyntheticSeries) -> None:
    rows = [
        {"t": index, "x": value, "series_name": series.name, "noise_level": series.noise_level}
        for index, value in enumerate(series.values)
    ]
    write_rows_csv(path, rows, ["t", "x", "series_name", "noise_level"])


def compute_ami(series_name: str, z_values: list[float], train_end: int) -> list[dict[str, Any]]:
    train = z_values[:train_end]
    edges = quantile_edges(train, AMI_BINS)
    binned = discretize(train, edges)
    rows = average_mutual_information_from_bins(binned, AMI_MAX_LAG, AMI_BINS)
    return [
        {"series_name": series_name, "tau": int(row["tau"]), "ami": float(row["mutual_information"])}
        for row in rows
    ]


def select_tau_operational(rows: list[dict[str, Any]]) -> tuple[int, str]:
    values = [float(row["ami"]) for row in rows]
    taus = [int(row["tau"]) for row in rows]
    for index in range(1, len(values) - 1):
        if values[index] < values[index - 1] and values[index] <= values[index + 1]:
            return taus[index], "primer minimo local de AMI"
    return 1, "sin minimo local claro; tau=1 operativo"


def select_embedding_dimension(
    m_fnn: int,
    m_cao: int,
    m_fnn_note: str,
    m_cao_note: str,
) -> tuple[int, str]:
    if m_fnn <= 2:
        return max(1, m_fnn), f"se prioriza FNN ({m_fnn_note}); Cao={m_cao} ({m_cao_note})"
    if abs(m_fnn - m_cao) <= 2:
        return max(1, min(MAX_DIM, m_fnn)), "FNN y Cao son compatibles; se usa FNN"
    return max(1, min(MAX_DIM, m_fnn)), f"FNN y Cao discrepan; se usa FNN={m_fnn} de forma operativa"


def run_lyapunov(
    series_name: str,
    z_values: list[float],
    tau: int,
    dim: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices = list(range((dim - 1) * tau, len(z_values)))
    vectors, _, _ = build_embedding_rows(z_values, [str(index) for index in range(len(z_values))], indices, tau, dim)
    block, block_start = contiguous_block(vectors, LYAP_BLOCK_SIZE, location="middle")
    theiler = max(1, tau * dim)
    curve, n_neighbors = rosenstein_curve(block, theiler, LYAP_K_MAX, series_name)
    rows = [
        {
            "series_name": series_name,
            "tau": tau,
            "m": dim,
            "k": int(row["k"]),
            "mean_log_distance": float(row["mean_log_distance"]),
            "n_pairs": int(row["n_pairs"]),
            "theiler_window": theiler,
        }
        for row in curve
    ]
    summary = summarize_lyapunov(curve, LYAP_FIT_START, LYAP_FIT_END)
    return rows, {
        "series_name": series_name,
        "tau": tau,
        "m": dim,
        "slope_per_step": float(summary["slope_per_5min_step"]),
        "fit_start": LYAP_FIT_START,
        "fit_end": LYAP_FIT_END,
        "r2_fit": float(summary["r_squared"]),
        "theoretical_ln2": math.log(2.0),
        "n_fit_points": int(summary["n_fit_points"]),
        "n_neighbor_pairs": n_neighbors,
        "block_start_embedding_index": block_start,
    }


def run_prediction(
    series_name: str,
    values: list[float],
    z_values: list[float],
    train_mean: float,
    train_std: float,
    tau: int,
    dim: int,
    train_end: int,
    validation_end: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    splits = build_prediction_splits(series_name, values, z_values, tau, dim, train_end, validation_end)
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    train_val = combine_splits("train_validation", [train, validation])
    theiler = max(1, tau * dim)
    mean_y_train = mean_known_targets(train, train_end)
    val_positions = sample_positions(len(validation.vectors), len(validation.vectors))
    test_positions = sample_positions(len(test.vectors), len(test.vectors))
    ar_model, ar_rows = select_ar_reference_by_bic(
        series_name,
        z_values[:train_end],
        max_order=MAX_AR_ORDER,
    )
    val_neighbors = compute_neighbor_sets(
        validation,
        train,
        val_positions,
        K_MAX,
        theiler,
        HORIZON,
        progress_label=None,
    )
    y_val = [validation.targets[position] for position in val_positions]
    val_rows: list[dict[str, Any]] = []
    for k in K_GRID:
        pred = knn_mean_predictions(val_neighbors, k)
        metrics = evaluate_predictions(f"knn_mean_k{k}", "validation", y_val, pred, mean_y_train)
        val_rows.append(
            {
                "series_name": series_name,
                "tau": tau,
                "m": dim,
                "theiler_window": theiler,
                "k": k,
                "n": metrics.n,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2_oos": metrics.r2_oos,
            }
        )
    selected_k = int(min(val_rows, key=lambda row: float(row["rmse"]))["k"])
    test_neighbors = compute_neighbor_sets(
        test,
        train_val,
        test_positions,
        max(K_MAX, selected_k),
        theiler,
        HORIZON,
        progress_label=None,
    )
    y_test = [test.targets[position] for position in test_positions]
    test_indices = [test.indices[position] for position in test_positions]
    ar_predictions = ar_bic_predictions(ar_model, z_values, test_indices, train_mean, train_std, mean_y_train)
    predictions = {
        "historical_mean": [mean_y_train] * len(test_positions),
        "persistence": [test.persistence[position] for position in test_positions],
        "ar_bic": ar_predictions,
        "nearest_neighbor": nearest_neighbor_predictions(test_neighbors),
        f"knn_mean_k{selected_k}": knn_mean_predictions(test_neighbors, selected_k),
    }
    metric_rows: list[dict[str, Any]] = []
    for model, pred in predictions.items():
        metrics = metrics_to_row(evaluate_predictions(model, "test", y_test, pred, mean_y_train))
        metric_rows.append(
            {
                "series_name": series_name,
                "model": model,
                "split": "test",
                "tau": tau,
                "m": dim,
                "theiler_window": theiler,
                "selected_k": selected_k,
                "ar_order": ar_model.order if model == "ar_bic" else "",
                "n": metrics["n"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2_oos": metrics["r2_oos"],
            }
        )
    knn_row = next(row for row in metric_rows if row["model"] == f"knn_mean_k{selected_k}")
    persistence_row = next(row for row in metric_rows if row["model"] == "persistence")
    ar_row = next(row for row in metric_rows if row["model"] == "ar_bic")
    return val_rows, metric_rows, {
        "series_name": series_name,
        "tau": tau,
        "m": dim,
        "theiler_window": theiler,
        "selected_k": selected_k,
        "ar_order": ar_model.order,
        "max_ar_order_tested": MAX_AR_ORDER,
        "ar_order_candidates": f"0..{MAX_AR_ORDER}",
        "test_rmse_knn": knn_row["rmse"],
        "test_rmse_persistence": persistence_row["rmse"],
        "test_rmse_ar": ar_row["rmse"],
        "delta_rmse_knn_vs_persistence": knn_row["rmse"] - persistence_row["rmse"],
        "delta_rmse_knn_vs_ar": knn_row["rmse"] - ar_row["rmse"],
    }, ar_rows


def select_ar_reference_by_bic(
    series_name: str,
    standardized_train: list[float],
    max_order: int,
) -> tuple[ARModel, list[dict[str, Any]]]:
    """Selecciona la referencia lineal AR(p), p=0..max_order, usando solo train.

    AR(0) se define como prediccion constante mediante la media historica de train.
    La tabla devuelve la varianza de innovacion en la escala estandarizada de train.
    """
    if len(standardized_train) <= max_order + 1:
        raise ValueError("Entrenamiento demasiado corto para seleccionar AR")
    nobs = len(standardized_train)
    train_mean = sum(standardized_train) / nobs
    zero_residual_variance = sum((value - train_mean) ** 2 for value in standardized_train) / nobs
    zero_sigma2 = max(zero_residual_variance, 1e-300)

    rows: list[dict[str, Any]] = []

    def criterion_row(order: int, sigma2: float, params: int) -> dict[str, Any]:
        safe_sigma2 = max(sigma2, 1e-300)
        return {
            "series_name": series_name,
            "p": order,
            "nobs_train": nobs,
            "innovation_variance": safe_sigma2,
            "aic": nobs * math.log(safe_sigma2) + 2.0 * params,
            "bic": nobs * math.log(safe_sigma2) + math.log(nobs) * params,
        }

    zero_row = criterion_row(order=0, sigma2=zero_sigma2, params=1)
    rows.append(zero_row)

    _, raw_models = select_ar_yule_walker(standardized_train, max_order=max_order)
    adjusted_models: dict[int, ARModel] = {}
    for model in raw_models:
        # select_ar_yule_walker trabaja con ACF normalizada. Multiplicar por la
        # varianza AR(0) mantiene la comparabilidad absoluta con la media.
        sigma2 = max(model.innovation_variance * zero_sigma2, 1e-300)
        row = criterion_row(order=model.order, sigma2=sigma2, params=model.order + 1)
        rows.append(row)
        adjusted_models[model.order] = ARModel(
            order=model.order,
            intercept=model.intercept,
            coefficients=model.coefficients,
            innovation_variance=sigma2,
            aic=float(row["aic"]),
            bic=float(row["bic"]),
            nobs=nobs,
        )

    selected_row = min(rows, key=lambda row: float(row["bic"]))
    selected_order = int(selected_row["p"])
    if selected_order == 0:
        return (
            ARModel(
                order=0,
                intercept=0.0,
                coefficients=[],
                innovation_variance=float(selected_row["innovation_variance"]),
                aic=float(selected_row["aic"]),
                bic=float(selected_row["bic"]),
                nobs=nobs,
            ),
            rows,
        )
    return adjusted_models[selected_order], rows


def ar_bic_predictions(
    ar_model: ARModel,
    z_values: list[float],
    query_indices: list[int],
    train_mean: float,
    train_std: float,
    mean_y_train: float,
) -> list[float]:
    if ar_model.order == 0:
        return [mean_y_train] * len(query_indices)
    return ar_recursive_forecast(
        z_values,
        query_indices,
        ar_model.coefficients,
        HORIZON,
        train_mean,
        train_std,
    )


def selected_ar_order_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for series_name in ["logistic_clean", "logistic_noise_small", "logistic_noise_moderate"]:
        candidates = [row for row in rows if row["series_name"] == series_name]
        if candidates:
            selected.append(min(candidates, key=lambda row: float(row["bic"])))
    return selected


def build_prediction_splits(
    series_name: str,
    values: list[float],
    z_values: list[float],
    tau: int,
    dim: int,
    train_end: int,
    validation_end: int,
) -> dict[str, EmbeddedSplit]:
    split_indices = {
        "train": list(range(0, min(train_end, len(values) - HORIZON))),
        "validation": list(range(train_end, min(validation_end, len(values) - HORIZON))),
        "test": list(range(validation_end, len(values) - HORIZON)),
    }
    times = [str(index) for index in range(len(values))]
    splits: dict[str, EmbeddedSplit] = {}
    for name, indices in split_indices.items():
        vectors, vector_indices, vector_times = build_embedding_rows(z_values, times, indices, tau, dim)
        splits[name] = EmbeddedSplit(
            name=name,
            vectors=vectors,
            indices=vector_indices,
            times=vector_times,
            targets=[values[index + HORIZON] for index in vector_indices],
            persistence=[values[index] for index in vector_indices],
        )
    return splits


def mean_known_targets(split: EmbeddedSplit, first_query_index: int) -> float:
    known = [
        target
        for index, target in zip(split.indices, split.targets)
        if index + HORIZON <= first_query_index
    ]
    return sum(known) / len(known)


def split_points(n: int) -> tuple[int, int]:
    train_end = int(0.60 * n)
    validation_end = int(0.80 * n)
    return train_end, validation_end


def write_embedding_figures(figures_dir: Path, embeddings: dict[str, dict[str, Any]]) -> None:
    for series_name, params in embeddings.items():
        z_values = params["z_values"]
        tau = int(params["tau"])
        indices = list(range(2 * tau, len(z_values)))
        times = [str(index) for index in range(len(z_values))]
        vectors2, vector_indices2, _ = build_embedding_rows(z_values, times, indices, tau, 2)
        vectors3, vector_indices3, _ = build_embedding_rows(z_values, times, indices, tau, 3)
        write_embedding_2d_svg(
            figures_dir / f"phase13_embedding_2d_{suffix(series_name)}.svg",
            vectors2,
            vector_indices2,
            f"Embedding 2D {series_name}: tau={tau}",
        )
        write_embedding_3d_svg(
            figures_dir / f"phase13_embedding_3d_{suffix(series_name)}.svg",
            vectors3,
            vector_indices3,
            f"Embedding 3D {series_name}: tau={tau}",
        )


def write_prediction_real_vs_predicted_svg(
    path: Path,
    series_name: str,
    metric_rows: list[dict[str, Any]],
    embedding_params: dict[str, Any],
    values: list[float],
    compact: bool = False,
    time_window: tuple[int, int] | None = None,
) -> None:
    tau = int(embedding_params["tau"])
    dim = int(embedding_params["m"])
    train_end = int(embedding_params["train_end"])
    validation_end = int(embedding_params["validation_end"])
    train_mean = float(embedding_params["train_mean"])
    train_std = float(embedding_params["train_std"])
    z_values = embedding_params["z_values"]
    splits = build_prediction_splits(series_name, values, z_values, tau, dim, train_end, validation_end)
    train_val = combine_splits("train_validation", [splits["train"], splits["validation"]])
    test = splits["test"]
    theiler = max(1, tau * dim)
    mean_y_train = mean_known_targets(splits["train"], train_end)
    selected_k = int(next(row for row in metric_rows if row["series_name"] == series_name and row["model"].startswith("knn_mean"))["selected_k"])
    if time_window is not None:
        window_start, window_end = time_window
        positions = [
            position
            for position, index in enumerate(test.indices)
            if window_start <= index < window_end
        ]
        if not positions:
            raise ValueError(
                f"Ventana temporal vacia para {series_name}: "
                f"[{window_start}, {window_end})"
            )
    elif compact:
        positions = continuous_block_positions(len(test.vectors), min(180, len(test.vectors)), center_fraction=0.5)
    else:
        positions = sample_positions(len(test.vectors), min(500, len(test.vectors)))
    neighbors = compute_neighbor_sets(test, train_val, positions, selected_k, theiler, HORIZON)
    ar_model, _ = select_ar_reference_by_bic(
        series_name,
        z_values[:train_end],
        max_order=MAX_AR_ORDER,
    )
    indices = [test.indices[position] for position in positions]
    ar_predictions = ar_bic_predictions(ar_model, z_values, indices, train_mean, train_std, mean_y_train)
    knn = knn_mean_predictions(neighbors, selected_k)
    times = [test.times[position] for position in positions]
    series = [
        ("real", [test.targets[position] for position in positions], "#222222"),
        ("persistencia", [test.persistence[position] for position in positions], "#6f7f8f"),
        (ar_display_label(ar_model.order), ar_predictions, "#2a6fbb"),
        (f"kNN k={selected_k}", knn, "#b45f06"),
    ]
    if time_window is not None:
        title_suffix = f"corta t={times[0]}-{times[-1]}"
    else:
        title_suffix = "compacta" if compact else "completa"
    write_time_series_svg(path, times, series, f"Prediccion test {title_suffix} {series_name}", "x[t+1]")


def prediction_short_windows(n_values: int) -> list[tuple[int, int]]:
    _, validation_end = split_points(n_values)
    test_start = validation_end
    test_end = n_values - HORIZON
    test_length = test_end - test_start
    window_size = min(PREDICTION_SHORT_WINDOW_SIZE, test_length)
    windows: list[tuple[int, int]] = []
    for center_fraction in PREDICTION_SHORT_WINDOW_CENTER_FRACTIONS:
        center = test_start + round((test_length - 1) * center_fraction)
        window_start = center - window_size // 2
        window_start = max(test_start, min(window_start, test_end - window_size))
        windows.append((window_start, window_start + window_size))
    return windows


def write_series_examples_svg(path: Path, series_list: list[SyntheticSeries]) -> None:
    rows = []
    for series in series_list:
        start = 1200
        rows.append((series.name, [str(i) for i in range(start, start + 400)], series.values[start:start + 400]))
    write_multi_panel_svg(path, rows, "Mapa logistico: ejemplos de series", "x")


def write_ami_svg(path: Path, rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> None:
    selected = {row["series_name"]: int(row["tau_selected"]) for row in selected_rows}
    groups = ["logistic_clean", "logistic_noise_small", "logistic_noise_moderate"]
    colors = {
        "logistic_clean": "#2a6fbb",
        "logistic_noise_small": "#6b8e23",
        "logistic_noise_moderate": "#b45f06",
    }
    labels = {
        "logistic_clean": "serie limpia",
        "logistic_noise_small": "ruido pequeno",
        "logistic_noise_moderate": "ruido moderado",
    }
    grouped_rows = {
        group: sorted(
            [row for row in rows if row["series_name"] == group],
            key=lambda row: int(row["tau"]),
        )
        for group in groups
    }

    width, height = 980, 480
    margin = {"left": 76, "right": 28, "top": 58, "bottom": 62}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    x_min = min(int(row["tau"]) for row in rows)
    x_max = max(int(row["tau"]) for row in rows)
    y_min = 0.0
    y_max = 1.08 * max(float(row["ami"]) for row in rows)

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(
        width,
        height,
        margin,
        "AMI del mapa logistico",
        y_min,
        y_max,
        y_coord,
        plot_width,
        plot_height,
    )
    elements.extend(x_axis_ticks(margin, plot_width, plot_height, x_min, x_max, x_coord, count=6))

    for group in groups:
        points = " ".join(
            f"{x_coord(int(row['tau'])):.2f},{y_coord(float(row['ami'])):.2f}"
            for row in grouped_rows[group]
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[group]}" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )

    for tau, label, color, label_offset in [
        (5, "tau=5 (ruido moderado)", colors["logistic_noise_moderate"], 15),
        (9, "tau=9 (limpia y ruido pequeno)", "#4d6f45", 31),
    ]:
        x = x_coord(tau)
        elements.append(
            f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" '
            f'y2="{margin["top"] + plot_height:.2f}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="5,4"/>'
        )
        elements.append(
            f'<text x="{x + 4:.2f}" y="{margin["top"] + label_offset:.2f}" '
            f'font-family="Arial, sans-serif" font-size="10" fill="{color}">{esc(label)}</text>'
        )

    for group in groups:
        tau = selected[group]
        selected_row = next(row for row in grouped_rows[group] if int(row["tau"]) == tau)
        elements.append(
            f'<circle cx="{x_coord(tau):.2f}" cy="{y_coord(float(selected_row["ami"])):.2f}" '
            f'r="4" fill="{colors[group]}" stroke="#ffffff" stroke-width="1.2"/>'
        )

    for index, group in enumerate(groups):
        elements.append(legend_item(width - 225, 28 + 18 * index, colors[group], labels[group]))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "tau", "AMI"))

    zoom_start, zoom_end = 5, 12
    zoom_rows = [
        row
        for row in rows
        if zoom_start <= int(row["tau"]) <= zoom_end
    ]
    zoom_values = [float(row["ami"]) for row in zoom_rows]
    zoom_y_min, zoom_y_max = expanded_range(min(zoom_values), max(zoom_values), 0.10)
    inset = {"left": 555.0, "top": 118.0, "width": 380.0, "height": 205.0}

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
        "Detalle de los primeros minimos locales</text>"
    )
    for tick in ticks(zoom_y_min, zoom_y_max, 4):
        y = inset_y(tick)
        elements.append(
            f'<line x1="{inset["left"]}" y1="{y:.2f}" '
            f'x2="{inset["left"] + inset["width"]:.2f}" y2="{y:.2f}" '
            f'stroke="#eeeeee" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{inset["left"] - 5:.2f}" y="{y + 3:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="8">{tick:.3f}</text>'
        )
    for tau, color in [(5, colors["logistic_noise_moderate"]), (9, "#4d6f45")]:
        x = inset_x(tau)
        elements.append(
            f'<line x1="{x:.2f}" y1="{inset["top"] + 24:.2f}" x2="{x:.2f}" '
            f'y2="{inset["top"] + inset["height"]:.2f}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="4,3"/>'
        )
    for group in groups:
        group_zoom = [
            row
            for row in grouped_rows[group]
            if zoom_start <= int(row["tau"]) <= zoom_end
        ]
        points = " ".join(
            f"{inset_x(int(row['tau'])):.2f},{inset_y(float(row['ami'])):.2f}"
            for row in group_zoom
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[group]}" '
            f'stroke-width="1.8" stroke-linejoin="round"/>'
        )
        for row in group_zoom:
            tau = int(row["tau"])
            is_selected = tau == selected[group]
            elements.append(
                f'<circle cx="{inset_x(tau):.2f}" cy="{inset_y(float(row["ami"])):.2f}" '
                f'r="{3.8 if is_selected else 2.0}" fill="{colors[group]}"/>'
            )
    for tick in [5, 9, 12]:
        elements.append(
            f'<text x="{inset_x(tick):.2f}" y="{inset["top"] + inset["height"] + 13:.2f}" '
            f'text-anchor="middle" font-family="Arial, sans-serif" font-size="9">{tick}</text>'
        )
    elements.append(
        f'<text x="{inset["left"] + inset["width"]/2:.2f}" '
        f'y="{inset["top"] + inset["height"] + 27:.2f}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="9">'
        "Se selecciona el primer minimo local, no el minimo global</text>"
    )

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_fnn_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    write_grouped_line_svg(
        path,
        rows,
        "m",
        "fnn_percent",
        "Falsos vecinos",
        "m",
        "FNN (%)",
        y_limits=(0.0, 100.0),
    )


def write_cao_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    e1_rows = [{**row, "value": row["E1"]} for row in rows]
    write_grouped_line_svg(path, e1_rows, "m", "value", "Metodo de Cao: E1", "m", "E1")


def write_lyapunov_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    write_grouped_line_svg(path, rows, "k", "mean_log_distance", "Rosenstein: divergencia media", "k", "log distancia media")


def write_prediction_k_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    write_grouped_line_svg(path, rows, "k", "rmse", "Prediccion local: RMSE validation por k", "k", "RMSE")


def write_ar_bic_selection_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = {
        row["series_name"]: int(row["p"])
        for row in selected_ar_order_rows(rows)
    }
    write_grouped_line_svg(
        path,
        rows,
        "p",
        "bic",
        "Referencia lineal: seleccion BIC de AR(p)",
        "p",
        "BIC",
        selected_tau=selected,
    )


def write_transition_map_clean_svg(path: Path, values: list[float], train_end: int) -> None:
    pairs = list(zip(values[:-1], values[1:]))
    positions = sample_positions(len(pairs), min(3500, len(pairs)))
    sampled = [pairs[position] for position in positions]
    train_pairs = pairs[: max(2, min(train_end - 1, len(pairs)))]
    train_x = [pair[0] for pair in train_pairs]
    train_y = [pair[1] for pair in train_pairs]
    slope, intercept = linear_fit(train_x, train_y)

    width, height = 760, 640
    margin = {"left": 76, "right": 32, "top": 58, "bottom": 64}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, "Mapa de transicion limpio: x[t] -> x[t+1]", y_min, y_max, y_coord, plot_width, plot_height)
    for tick in ticks(x_min, x_max, 6):
        x = x_coord(tick)
        elements.append(f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" y2="{margin["top"] + plot_height:.2f}" stroke="#eeeeee"/>')
        elements.append(f'<text x="{x:.2f}" y="{margin["top"] + plot_height + 18:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">{tick:.2g}</text>')
    for x, y in sampled:
        elements.append(f'<circle cx="{x_coord(x):.2f}" cy="{y_coord(y):.2f}" r="1.35" fill="#2a6fbb" opacity="0.32"/>')

    theoretical_points = " ".join(
        f"{x_coord(x):.2f},{y_coord(4.0 * x * (1.0 - x)):.2f}"
        for x in [index / 200.0 for index in range(201)]
    )
    elements.append(f'<polyline points="{theoretical_points}" fill="none" stroke="#b45f06" stroke-width="2.1" stroke-dasharray="8,5"/>')
    line_points = " ".join(
        f"{x_coord(x):.2f},{y_coord(intercept + slope * x):.2f}"
        for x in [0.0, 1.0]
    )
    elements.append(f'<polyline points="{line_points}" fill="none" stroke="#6f7f8f" stroke-width="2" stroke-dasharray="6,4"/>')
    elements.append(legend_item(width - 210, 28, "#2a6fbb", "pares observados"))
    elements.append(legend_line(width - 210, 48, "#b45f06", "4x(1-x) referencia", dasharray="8,5"))
    elements.append(legend_line(width - 210, 68, "#6f7f8f", "ajuste lineal train", dasharray="6,4"))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "x[t]", "x[t+1]"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def write_prediction_metrics_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    series_names = ["logistic_clean", "logistic_noise_small", "logistic_noise_moderate"]
    models = ["historical_mean", "persistence", "ar_bic", "nearest_neighbor"]
    knn_models = {
        name: next(row["model"] for row in rows if row["series_name"] == name and str(row["model"]).startswith("knn_mean"))
        for name in series_names
    }
    ar_orders = {
        int(float(row["ar_order"]))
        for row in rows
        if row["model"] == "ar_bic" and str(row.get("ar_order", "")).strip()
    }
    ar_label = ar_display_label(next(iter(ar_orders))) if len(ar_orders) == 1 else "AR(p BIC)"
    labels = models + ["knn_selected"]
    width, height = 1120, 520
    margin = {"left": 76, "right": 30, "top": 58, "bottom": 130}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    values: list[tuple[str, str, float]] = []
    for series_name in series_names:
        for label in labels:
            model = knn_models[series_name] if label == "knn_selected" else label
            row = next(item for item in rows if item["series_name"] == series_name and item["model"] == model)
            values.append((series_name, label, float(row["rmse"])))
    y_min, y_max = 0.0, max(value for _, _, value in values) * 1.12

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, "Test: RMSE por serie y modelo", y_min, y_max, y_coord, plot_width, plot_height)
    colors = {
        "historical_mean": "#6f7f8f",
        "persistence": "#6b8e23",
        "ar_bic": "#2a6fbb",
        "nearest_neighbor": "#8a5fbf",
        "knn_selected": "#b45f06",
    }
    display_labels = {
        "historical_mean": "media historica",
        "persistence": "persistencia",
        "ar_bic": ar_label,
        "nearest_neighbor": "vecino 1",
        "knn_selected": "kNN seleccionado",
    }
    group_width = plot_width / len(series_names)
    bar_width = group_width / (len(labels) + 3.0)
    for group_index, series_name in enumerate(series_names):
        center = margin["left"] + group_width * (group_index + 0.5)
        for model_index, label in enumerate(labels):
            row = next(item for item in values if item[0] == series_name and item[1] == label)
            bar_center = center + (model_index - (len(labels) - 1) / 2.0) * bar_width
            x = bar_center - bar_width / 2.0
            y = y_coord(row[2])
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width*0.82:.2f}" '
                f'height="{margin["top"] + plot_height - y:.2f}" fill="{colors[label]}" opacity="0.78"/>'
            )
        elements.append(
            f'<text x="{center:.2f}" y="{height - 74}" text-anchor="end" '
            f'transform="rotate(-28 {center:.2f},{height - 74})" font-family="Arial, sans-serif" font-size="11">{esc(series_name)}</text>'
        )
    for index, label in enumerate(labels):
        elements.append(legend_item(width - 190, 28 + 18 * index, colors[label], display_labels[label]))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "serie", "RMSE"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_embedding_2d_svg(path: Path, vectors: list[list[float]], indices: list[int], title: str) -> None:
    selected = select_points(vectors, indices, 3500)
    xs = [row[0][0] for row in selected]
    ys = [row[0][1] for row in selected]
    write_scatter_svg(path, xs, ys, [row[1] for row in selected], title, "x_t", "x_t-tau")


def write_embedding_3d_svg(path: Path, vectors: list[list[float]], indices: list[int], title: str) -> None:
    selected = select_points(vectors, indices, 3500)
    projected_x: list[float] = []
    projected_y: list[float] = []
    colors: list[int] = []
    for vector, index in selected:
        x, y, z = vector[0], vector[1], vector[2]
        projected_x.append(x - 0.55 * y)
        projected_y.append(0.35 * x + 0.35 * y - z)
        colors.append(index)
    write_scatter_svg(path, projected_x, projected_y, colors, title, "proyeccion 3D x", "proyeccion 3D y")


def write_scatter_svg(
    path: Path,
    xs: list[float],
    ys: list[float],
    color_values: list[int],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 720, 620
    margin = {"left": 76, "right": 28, "top": 58, "bottom": 62}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    x_min, x_max = expanded_range(min(xs), max(xs), 0.06)
    y_min, y_max = expanded_range(min(ys), max(ys), 0.06)
    c_min, c_max = min(color_values), max(color_values)

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    for x, y, color_value in zip(xs, ys, color_values):
        frac = (color_value - c_min) / max(1, c_max - c_min)
        color = interpolate_color("#2a6fbb", "#b45f06", frac)
        elements.append(f'<circle cx="{x_coord(x):.2f}" cy="{y_coord(y):.2f}" r="1.45" fill="{color}" opacity="0.62"/>')
    elements.extend(x_axis_ticks(margin, plot_width, plot_height, x_min, x_max, x_coord, count=5))
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, x_label, y_label))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_multi_panel_svg(
    path: Path,
    panels: list[tuple[str, list[str], list[float]]],
    title: str,
    y_label: str,
) -> None:
    width, height = 1240, 660
    margin = {"left": 204, "right": 28, "top": 60, "bottom": 72}
    gap = 34
    panel_height = (height - margin["top"] - margin["bottom"] - gap * (len(panels) - 1)) / len(panels)
    plot_width = width - margin["left"] - margin["right"]
    elements = [
        svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700">{esc(title)}</text>',
    ]
    for panel_index, (label, times, values) in enumerate(panels):
        top = margin["top"] + panel_index * (panel_height + gap)
        y_min, y_max = expanded_range(min(values), max(values), 0.08)

        def x_coord(index: int) -> float:
            return margin["left"] + plot_width * index / max(1, len(values) - 1)

        def y_coord(value: float) -> float:
            return top + panel_height - panel_height * (value - y_min) / (y_max - y_min)

        elements.append(f'<rect x="{margin["left"]}" y="{top:.2f}" width="{plot_width:.2f}" height="{panel_height:.2f}" fill="none" stroke="#222222"/>')
        for tick in ticks(y_min, y_max, 3):
            y = y_coord(tick)
            elements.append(f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_width:.2f}" y2="{y:.2f}" stroke="#eeeeee"/>')
            elements.append(f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10">{tick:.3g}</text>')
        points = " ".join(f"{x_coord(index):.2f},{y_coord(value):.2f}" for index, value in enumerate(values))
        elements.append(f'<polyline points="{points}" fill="none" stroke="#2a6fbb" stroke-width="1.3"/>')
        elements.append(f'<text x="{margin["left"] - 12}" y="{top + 16:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{esc(label)}</text>')
        if panel_index == len(panels) - 1:
            bottom = top + panel_height
            for position in sorted(set([0, len(values) // 4, len(values) // 2, 3 * len(values) // 4, len(values) - 1])):
                x = x_coord(position)
                elements.append(f'<line x1="{x:.2f}" y1="{bottom:.2f}" x2="{x:.2f}" y2="{bottom + 5:.2f}" stroke="#222222"/>')
                elements.append(f'<text x="{x:.2f}" y="{bottom + 18:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{esc(times[position])}</text>')
    elements.append(f'<text transform="translate(24,{height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(y_label)}</text>')
    elements.append(f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">t</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_grouped_line_svg(
    path: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    selected_tau: dict[str, int] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    width, height = 980, 460
    margin = {"left": 76, "right": 28, "top": 58, "bottom": 62}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    groups = ["logistic_clean", "logistic_noise_small", "logistic_noise_moderate"]
    colors = {"logistic_clean": "#2a6fbb", "logistic_noise_small": "#6b8e23", "logistic_noise_moderate": "#b45f06"}
    finite_rows = [row for row in rows if math.isfinite(float(row[y_key]))]
    x_values = [float(row[x_key]) for row in finite_rows]
    y_values = [float(row[y_key]) for row in finite_rows]
    x_min, x_max = min(x_values), max(x_values)
    if y_limits is None:
        y_min, y_max = expanded_range(min(y_values), max(y_values), 0.08)
    else:
        y_min, y_max = y_limits
        if y_min >= y_max:
            raise ValueError("Los limites del eje Y deben cumplir y_min < y_max")

    def x_coord(value: float) -> float:
        return margin["left"] + plot_width * (value - x_min) / max(1e-12, x_max - x_min)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    for group in groups:
        group_rows = sorted([row for row in finite_rows if row["series_name"] == group], key=lambda row: float(row[x_key]))
        points = " ".join(f"{x_coord(float(row[x_key])):.2f},{y_coord(float(row[y_key])):.2f}" for row in group_rows)
        elements.append(f'<polyline points="{points}" fill="none" stroke="{colors[group]}" stroke-width="2"/>')
        if selected_tau and group in selected_tau:
            x = x_coord(selected_tau[group])
            elements.append(f'<line x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" y2="{margin["top"] + plot_height:.2f}" stroke="{colors[group]}" stroke-width="1" stroke-dasharray="4,4"/>')
    for index, group in enumerate(groups):
        elements.append(legend_item(width - 250, 28 + 18 * index, colors[group], group))
    elements.extend(x_axis_ticks(margin, plot_width, plot_height, x_min, x_max, x_coord, count=6))
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
    width, height = 1080, 460
    margin = {"left": 76, "right": 30, "top": 58, "bottom": 62}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    all_y = [value for _, values, _ in series for value in values]
    y_min, y_max = expanded_range(min(all_y), max(all_y), 0.08)
    n = len(times)

    def x_coord(index: int) -> float:
        return margin["left"] + plot_width * index / max(1, n - 1)

    def y_coord(value: float) -> float:
        return margin["top"] + plot_height - plot_height * (value - y_min) / (y_max - y_min)

    elements = base_svg(width, height, margin, title, y_min, y_max, y_coord, plot_width, plot_height)
    for label, values, color in series:
        points = " ".join(f"{x_coord(index):.2f},{y_coord(value):.2f}" for index, value in enumerate(values))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    for index, (label, _, color) in enumerate(series):
        elements.append(legend_line(width - 210, 28 + 18 * index, color, label))
    for position in sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1])):
        label = times[min(position, len(times) - 1)]
        elements.append(f'<line x1="{x_coord(position):.2f}" y1="{margin["top"] + plot_height:.2f}" x2="{x_coord(position):.2f}" y2="{margin["top"] + plot_height + 5:.2f}" stroke="#222222"/>')
        elements.append(f'<text x="{x_coord(position):.2f}" y="{height - 38}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{esc(label)}</text>')
    elements.append(axis_labels(width, height, margin, plot_width, plot_height, "t", y_label))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")




def prediction_for(series_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["series_name"] == series_name and str(row["model"]).startswith("knn_mean")]
    return candidates[0]


def metric_for(series_name: str, model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in rows if row["series_name"] == series_name and row["model"] == model)


def print_final_summary(
    selected_rows: list[dict[str, Any]],
    lyapunov_rows: list[dict[str, Any]],
    prediction_summary: list[dict[str, Any]],
) -> None:
    print("Resumen Fase 13")
    for row in selected_rows:
        print(f"{row['series_name']}: tau={row['tau_selected']}, m={row['m_selected']}")
    for row in lyapunov_rows:
        print(f"{row['series_name']}: lyap slope={float(row['slope_per_step']):.6g}, R2={float(row['r2_fit']):.4g}")
    for row in prediction_summary:
        print(
            f"{row['series_name']}: "
            f"{ar_display_label(int(row['ar_order']))} seleccionado por BIC, "
            f"RMSE AR={float(row['test_rmse_ar']):.6g}, "
            f"k={row['selected_k']}, RMSE kNN={float(row['test_rmse_knn']):.6g}, "
            f"delta kNN vs AR={float(row['delta_rmse_knn_vs_ar']):.6g}"
        )
    print("Validacion sintetica completada.")


def select_points(vectors: list[list[float]], indices: list[int], max_points: int) -> list[tuple[list[float], int]]:
    if len(vectors) <= max_points:
        return list(zip(vectors, indices))
    positions = sample_positions(len(vectors), max_points)
    return [(vectors[position], indices[position]) for position in positions]


def suffix(series_name: str) -> str:
    return series_name.replace("logistic_", "")


def ar_display_label(order: int) -> str:
    if order == 0:
        return "AR(0) / media"
    return f"AR({order})"


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


def axis_labels(width: int, height: int, margin: dict[str, int], plot_width: float, plot_height: float, x_label: str, y_label: str) -> str:
    return "\n".join([
        f'<text x="{margin["left"] + plot_width/2:.2f}" y="{height - 10}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(x_label)}</text>',
        f'<text transform="translate(18,{margin["top"] + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{esc(y_label)}</text>',
    ])


def legend_item(x: float, y: float, color: str, label: str) -> str:
    return f'<g><circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/><text x="{x + 10:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" font-size="12">{esc(label)}</text></g>'


def legend_line(
    x: float,
    y: float,
    color: str,
    label: str,
    dasharray: str | None = None,
) -> str:
    dash_style = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return f'<g><line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 28:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2"{dash_style}/><text x="{x + 36:.2f}" y="{y + 4:.2f}" font-family="Arial, sans-serif" font-size="12">{esc(label)}</text></g>'




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


def interpolate_color(left: str, right: str, fraction: float) -> str:
    l_values = [int(left[index:index + 2], 16) for index in (1, 3, 5)]
    r_values = [int(right[index:index + 2], 16) for index in (1, 3, 5)]
    values = [round(l + (r - l) * fraction) for l, r in zip(l_values, r_values)]
    return "#" + "".join(f"{value:02x}" for value in values)


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ejecuta Fase 13: mapa logistico.")
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
