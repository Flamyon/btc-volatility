"""Prediccion local en espacio de estados para la Fase 11."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import heapq
import math
from pathlib import Path
from typing import Any

from state_space import build_embedding_rows, standardize_train


TARGET_HORIZON_BARS = 12


@dataclass(frozen=True)
class SeriesData:
    times: list[str]
    x: list[float]
    y: list[float]


@dataclass(frozen=True)
class EmbeddedSplit:
    name: str
    vectors: list[list[float]]
    indices: list[int]
    times: list[str]
    targets: list[float]
    persistence: list[float]


@dataclass(frozen=True)
class NeighborSet:
    distances: list[float]
    targets: list[float]
    indices: list[int]


@dataclass(frozen=True)
class Metrics:
    model: str
    split: str
    n: int
    mae: float
    mse: float
    rmse: float
    r2_oos: float
    bias: float
    error_std: float


class KDNode:
    __slots__ = ("point", "axis", "left", "right")

    def __init__(
        self,
        point: int,
        axis: int,
        left: "KDNode | None",
        right: "KDNode | None",
    ) -> None:
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """KD-tree exacto para k vecinos en baja dimension."""

    def __init__(self, vectors: list[list[float]]) -> None:
        if not vectors:
            raise ValueError("No se puede construir KDTree vacio")
        self.vectors = vectors
        self.dim = len(vectors[0])
        self.root = self._build(list(range(len(vectors))), depth=0)

    def query(
        self,
        vector: list[float],
        k: int,
        query_index: int,
        candidate_indices: list[int],
        theiler_window: int,
        horizon: int,
    ) -> list[tuple[float, int]]:
        heap: list[tuple[float, int]] = []

        def eligible(candidate_position: int) -> bool:
            candidate_index = candidate_indices[candidate_position]
            if candidate_index + horizon > query_index:
                return False
            return abs(query_index - candidate_index) > theiler_window

        def visit(node: KDNode | None) -> None:
            if node is None:
                return
            point = node.point
            axis = node.axis
            point_vector = self.vectors[point]
            diff_axis = vector[axis] - point_vector[axis]
            near = node.left if diff_axis <= 0.0 else node.right
            far = node.right if diff_axis <= 0.0 else node.left

            visit(near)

            current_limit = -heap[0][0] if len(heap) >= k else float("inf")
            if eligible(point):
                distance_sq = distance_sq_bounded(vector, point_vector, current_limit)
                if distance_sq < current_limit or len(heap) < k:
                    item = (-distance_sq, -point)
                    if len(heap) < k:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heapreplace(heap, item)

            current_limit = -heap[0][0] if len(heap) >= k else float("inf")
            if len(heap) < k or diff_axis * diff_axis < current_limit:
                visit(far)

        visit(self.root)
        rows = [(-distance_sq, -position) for distance_sq, position in heap]
        rows.sort(key=lambda item: item[0])
        return rows

    def _build(self, points: list[int], depth: int) -> KDNode | None:
        if not points:
            return None
        axis = depth % self.dim
        points.sort(key=lambda point: self.vectors[point][axis])
        median = len(points) // 2
        return KDNode(
            point=points[median],
            axis=axis,
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1 :], depth + 1),
        )


def read_prediction_series(path: Path) -> SeriesData:
    times: list[str] = []
    x_values: list[float] = []
    y_values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"open_time", "log_rv_past_12", "log_rv_future_12"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
        for row in reader:
            x = float(row["log_rv_past_12"])
            y = float(row["log_rv_future_12"])
            if math.isfinite(x) and math.isfinite(y):
                times.append(row["open_time"])
                x_values.append(x)
                y_values.append(y)
    return SeriesData(times=times, x=x_values, y=y_values)


def split_row_indices(
    times: list[str],
    purge_bars: int = TARGET_HORIZON_BARS,
) -> dict[str, list[int]]:
    """Particiones cronologicas con purga del horizonte en cada frontera."""
    if purge_bars < 0:
        raise ValueError("purge_bars no puede ser negativo")
    splits = {"train": [], "validation": [], "test": []}
    for index, time in enumerate(times):
        if time <= "2025-06-30 23:55:00":
            splits["train"].append(index)
        elif time <= "2025-12-31 23:55:00":
            splits["validation"].append(index)
        elif time >= "2026-01-01 00:00:00":
            splits["test"].append(index)
    if purge_bars:
        for name in ("train", "validation"):
            if len(splits[name]) <= purge_bars:
                raise ValueError(f"Split {name} demasiado corto para purgar {purge_bars} velas")
            splits[name] = splits[name][:-purge_bars]
    return splits


def build_embedded_split(
    name: str,
    z_values: list[float],
    x_values: list[float],
    y_values: list[float],
    times: list[str],
    indices: list[int],
    tau: int,
    dim: int,
) -> EmbeddedSplit:
    vectors, vector_indices, vector_times = build_embedding_rows(
        z_values,
        times,
        indices,
        tau,
        dim,
    )
    targets = [y_values[index] for index in vector_indices]
    persistence = [x_values[index] for index in vector_indices]
    return EmbeddedSplit(
        name=name,
        vectors=vectors,
        indices=vector_indices,
        times=vector_times,
        targets=targets,
        persistence=persistence,
    )


def combine_splits(name: str, splits: list[EmbeddedSplit]) -> EmbeddedSplit:
    vectors: list[list[float]] = []
    indices: list[int] = []
    times: list[str] = []
    targets: list[float] = []
    persistence: list[float] = []
    for split in splits:
        vectors.extend(split.vectors)
        indices.extend(split.indices)
        times.extend(split.times)
        targets.extend(split.targets)
        persistence.extend(split.persistence)
    return EmbeddedSplit(name, vectors, indices, times, targets, persistence)


def sample_positions(length: int, sample_size: int) -> list[int]:
    if length <= 0:
        return []
    if length <= sample_size:
        return list(range(length))
    if sample_size <= 1:
        return [0]
    positions: list[int] = []
    previous = -1
    for output_index in range(sample_size):
        position = round((length - 1) * output_index / (sample_size - 1))
        if position != previous:
            positions.append(position)
            previous = position
    return positions


def continuous_block_positions(length: int, block_size: int, center_fraction: float = 0.5) -> list[int]:
    if length <= block_size:
        return list(range(length))
    center = int((length - 1) * center_fraction)
    start = max(0, min(length - block_size, center - block_size // 2))
    return list(range(start, start + block_size))


def compute_neighbor_sets(
    query_split: EmbeddedSplit,
    candidate_split: EmbeddedSplit,
    query_positions: list[int],
    k_max: int,
    theiler_window: int,
    horizon: int,
    progress_label: str | None = None,
    progress_every: int = 1000,
) -> list[NeighborSet]:
    tree = KDTree(candidate_split.vectors)
    rows: list[NeighborSet] = []
    for counter, position in enumerate(query_positions, start=1):
        neighbors = tree.query(
            query_split.vectors[position],
            k_max,
            query_split.indices[position],
            candidate_split.indices,
            theiler_window,
            horizon,
        )
        if not neighbors:
            raise ValueError("No se encontraron vecinos elegibles")
        rows.append(
            NeighborSet(
                distances=[math.sqrt(distance_sq) for distance_sq, _ in neighbors],
                targets=[candidate_split.targets[candidate_position] for _, candidate_position in neighbors],
                indices=[candidate_split.indices[candidate_position] for _, candidate_position in neighbors],
            )
        )
        if progress_label and (counter % progress_every == 0 or counter == len(query_positions)):
            print(f"{progress_label}: {counter}/{len(query_positions)}")
    return rows


def knn_mean_predictions(neighbor_sets: list[NeighborSet], k: int) -> list[float]:
    predictions: list[float] = []
    for neighbors in neighbor_sets:
        use_k = min(k, len(neighbors.targets))
        predictions.append(sum(neighbors.targets[:use_k]) / use_k)
    return predictions


def nearest_neighbor_predictions(neighbor_sets: list[NeighborSet]) -> list[float]:
    return [neighbors.targets[0] for neighbors in neighbor_sets]


def ar_recursive_forecast(
    z_values: list[float],
    query_indices: list[int],
    coefficients: list[float],
    horizon: int,
    x_mean_train: float,
    x_std_train: float,
) -> list[float]:
    predictions: list[float] = []
    order = len(coefficients)
    for query_index in query_indices:
        future: dict[int, float] = {}
        for step in range(1, horizon + 1):
            target_index = query_index + step
            forecast = 0.0
            for lag, coefficient in enumerate(coefficients, start=1):
                source_index = target_index - lag
                if source_index <= query_index:
                    source_value = z_values[source_index]
                else:
                    source_value = future[source_index]
                forecast += coefficient * source_value
            future[target_index] = forecast
        predictions.append(future[query_index + horizon] * x_std_train + x_mean_train)
    return predictions


def load_ar_coefficients(path: Path) -> list[float]:
    coeffs_by_lag: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lag = int(row["lag"])
            coeffs_by_lag[lag] = float(row["coefficient"])
    if not coeffs_by_lag:
        raise ValueError("No hay coeficientes AR")
    order = max(coeffs_by_lag)
    return [coeffs_by_lag[lag] for lag in range(1, order + 1)]


def evaluate_predictions(
    model: str,
    split: str,
    y_true: list[float],
    y_pred: list[float],
    mean_y_train: float,
) -> Metrics:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true/y_pred invalidos para metricas")
    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    mae = sum(abs_errors) / len(abs_errors)
    mse = sum(squared_errors) / len(squared_errors)
    rmse = math.sqrt(mse)
    denominator = sum((true - mean_y_train) ** 2 for true in y_true)
    r2_oos = 1.0 - sum(squared_errors) / denominator if denominator > 0.0 else float("nan")
    bias = sum(errors) / len(errors)
    error_std = sample_std(errors)
    return Metrics(model, split, len(y_true), mae, mse, rmse, r2_oos, bias, error_std)


def metrics_to_row(metrics: Metrics) -> dict[str, Any]:
    return {
        "model": metrics.model,
        "split": metrics.split,
        "n": metrics.n,
        "mae": metrics.mae,
        "mse": metrics.mse,
        "rmse": metrics.rmse,
        "r2_oos": metrics.r2_oos,
        "bias_yhat_minus_y": metrics.bias,
        "error_std": metrics.error_std,
    }


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def distance_sq_bounded(left: list[float], right: list[float], limit: float) -> float:
    distance = 0.0
    for left_value, right_value in zip(left, right):
        diff = left_value - right_value
        distance += diff * diff
        if distance >= limit:
            break
    return distance


def alignment_check(x_values: list[float], y_values: list[float], horizon: int) -> dict[str, Any]:
    diffs: list[float] = []
    for index in range(0, len(x_values) - horizon):
        diffs.append(abs(y_values[index] - x_values[index + horizon]))
    max_abs = max(diffs) if diffs else float("nan")
    mean_abs = sum(diffs) / len(diffs) if diffs else float("nan")
    return {
        "horizon": horizon,
        "n_checked": len(diffs),
        "max_abs_difference_y_t_vs_x_t_plus_horizon": max_abs,
        "mean_abs_difference_y_t_vs_x_t_plus_horizon": mean_abs,
    }


def prepare_phase11_splits(
    data: SeriesData,
    tau: int,
    dim: int,
) -> tuple[dict[str, EmbeddedSplit], list[float], float, float, dict[str, list[int]]]:
    row_splits = split_row_indices(data.times)
    train_end = sum(time <= "2025-06-30 23:55:00" for time in data.times)
    z_values, x_mean_train, x_std_train = standardize_train(data.x, train_end)
    splits = {
        name: build_embedded_split(
            name,
            z_values,
            data.x,
            data.y,
            data.times,
            indices,
            tau,
            dim,
        )
        for name, indices in row_splits.items()
    }
    return splits, z_values, x_mean_train, x_std_train, row_splits
