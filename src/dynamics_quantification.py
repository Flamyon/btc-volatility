"""Medidas basicas para cuantificar dinamica reconstruida."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import math
import random
from typing import Any


def select_evenly_spaced_pairs(
    vectors: list[list[float]],
    indices: list[int],
    sample_size: int,
) -> tuple[list[list[float]], list[int]]:
    """Submuestra equiespaciada de vectores e indices."""
    n = len(vectors)
    if n <= sample_size:
        return vectors[:], indices[:]
    selected_vectors: list[list[float]] = []
    selected_indices: list[int] = []
    for output_index in range(sample_size):
        source_index = round((n - 1) * output_index / (sample_size - 1))
        selected_vectors.append(vectors[source_index])
        selected_indices.append(indices[source_index])
    return selected_vectors, selected_indices


def contiguous_block(
    vectors: list[list[float]],
    block_size: int,
    location: str = "middle",
) -> tuple[list[list[float]], int]:
    """Bloque continuo para Rosenstein."""
    if len(vectors) <= block_size:
        return vectors[:], 0
    if location == "start":
        start = 0
    elif location == "end":
        start = len(vectors) - block_size
    else:
        start = len(vectors) // 2 - block_size // 2
    return vectors[start : start + block_size], start


def shuffled_scalar(values: list[float], seed: int) -> list[float]:
    """Copia barajada reproducible."""
    rng = random.Random(seed)
    shuffled = values[:]
    rng.shuffle(shuffled)
    return shuffled


def pairwise_distances_sorted(
    vectors: list[list[float]],
    indices: list[int],
    theiler_window: int,
) -> list[float]:
    """Distancias euclideas entre pares validos, excluyendo Theiler."""
    distances: list[float] = []
    n = len(vectors)
    for i in range(n - 1):
        vector_i = vectors[i]
        index_i = indices[i]
        for j in range(i + 1, n):
            if abs(index_i - indices[j]) <= theiler_window:
                continue
            distances.append(math.sqrt(euclidean_distance_sq(vector_i, vectors[j])))
    distances.sort()
    return distances


def radii_from_distances(
    sorted_distances: list[float],
    count: int,
    low_quantile: float = 0.01,
    high_quantile: float = 0.70,
) -> list[float]:
    """Radios log-espaciados entre cuantiles de distancias."""
    low = quantile(sorted_distances, low_quantile)
    high = quantile(sorted_distances, high_quantile)
    low = max(low, 1e-12)
    high = max(high, low * 1.001)
    log_low = math.log(low)
    log_high = math.log(high)
    return [
        math.exp(log_low + (log_high - log_low) * index / (count - 1))
        for index in range(count)
    ]


def correlation_curve_from_distances(
    sorted_distances: list[float],
    radii: list[float],
    series_label: str,
    n_vectors: int,
    theiler_window: int,
    metric: str = "euclidean",
) -> list[dict[str, Any]]:
    """Calcula C(r), log C(r) y pendiente local."""
    total_pairs = len(sorted_distances)
    rows: list[dict[str, Any]] = []
    for radius in radii:
        count = bisect_left(sorted_distances, radius)
        c_r = count / total_pairs if total_pairs else 0.0
        rows.append(
            {
                "series": series_label,
                "radius": radius,
                "log_radius": math.log(radius) if radius > 0.0 else float("nan"),
                "correlation_sum": c_r,
                "log_correlation_sum": math.log(c_r) if c_r > 0.0 else float("nan"),
                "local_slope": float("nan"),
                "n_vectors": n_vectors,
                "n_pairs": total_pairs,
                "metric": metric,
                "theiler_window": theiler_window,
            }
        )
    for index in range(1, len(rows) - 1):
        y_prev = float(rows[index - 1]["log_correlation_sum"])
        y_next = float(rows[index + 1]["log_correlation_sum"])
        x_prev = float(rows[index - 1]["log_radius"])
        x_next = float(rows[index + 1]["log_radius"])
        if math.isfinite(y_prev) and math.isfinite(y_next) and x_next != x_prev:
            rows[index]["local_slope"] = (y_next - y_prev) / (x_next - x_prev)
    return rows


def summarize_correlation_dimension(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen prudente de D2 a partir de pendientes locales."""
    candidates = [
        row for row in rows
        if 0.02 <= float(row["correlation_sum"]) <= 0.25
        and math.isfinite(float(row["local_slope"]))
        and float(row["local_slope"]) > 0.0
    ]
    if not candidates:
        return {
            "series": rows[0]["series"] if rows else "",
            "d2_estimate": None,
            "slope_std": None,
            "n_scaling_points": 0,
            "plateau_clear": False,
            "scaling_radius_min": None,
            "scaling_radius_max": None,
            "scaling_status": "insufficient_points",
        }
    slopes = [float(row["local_slope"]) for row in candidates]
    estimate = median(slopes)
    std = sample_std(slopes) if len(slopes) > 1 else 0.0
    plateau_clear = len(slopes) >= 5 and std <= max(0.20 * abs(estimate), 0.15)
    return {
        "series": candidates[0]["series"],
        "d2_estimate": estimate,
        "slope_std": std,
        "n_scaling_points": len(slopes),
        "plateau_clear": plateau_clear,
        "scaling_radius_min": min(float(row["radius"]) for row in candidates),
        "scaling_radius_max": max(float(row["radius"]) for row in candidates),
        "scaling_status": "stable_plateau" if plateau_clear else "unstable_plateau",
    }


def rosenstein_curve(
    vectors: list[list[float]],
    theiler_window: int,
    k_max: int,
    series_label: str,
) -> tuple[list[dict[str, Any]], int]:
    """Metodo de Rosenstein para divergencia media."""
    valid_n = len(vectors) - k_max
    if valid_n <= theiler_window + 2:
        raise ValueError("Bloque demasiado corto para k_max y Theiler")

    neighbors: list[tuple[int, int]] = []
    for i in range(valid_n):
        best_j = -1
        best_distance = float("inf")
        vector_i = vectors[i]
        for j in range(valid_n):
            if i == j or abs(i - j) <= theiler_window:
                continue
            distance = euclidean_distance_sq(vector_i, vectors[j])
            if distance < best_distance:
                best_distance = distance
                best_j = j
        if best_j >= 0:
            neighbors.append((i, best_j))

    rows: list[dict[str, Any]] = []
    for k in range(k_max + 1):
        log_distances: list[float] = []
        for i, j in neighbors:
            distance = math.sqrt(euclidean_distance_sq(vectors[i + k], vectors[j + k]))
            if distance > 1e-12:
                log_distances.append(math.log(distance))
        mean_log = sum(log_distances) / len(log_distances) if log_distances else float("nan")
        rows.append(
            {
                "series": series_label,
                "k": k,
                "time_minutes": 5 * k,
                "mean_log_distance": mean_log,
                "n_pairs": len(log_distances),
                "theiler_window": theiler_window,
            }
        )
    return rows, len(neighbors)


def summarize_lyapunov(
    rows: list[dict[str, Any]],
    fit_start: int,
    fit_end: int,
) -> dict[str, Any]:
    """Ajuste lineal de la curva Rosenstein."""
    selected = [
        row for row in rows
        if fit_start <= int(row["k"]) <= fit_end
        and math.isfinite(float(row["mean_log_distance"]))
    ]
    x = [float(row["k"]) for row in selected]
    y = [float(row["mean_log_distance"]) for row in selected]
    slope, intercept, r2 = linear_fit(x, y)
    return {
        "series": rows[0]["series"] if rows else "",
        "fit_start_k": fit_start,
        "fit_end_k": fit_end,
        "slope_per_5min_step": slope,
        "slope_per_hour": slope * 12.0,
        "intercept": intercept,
        "r_squared": r2,
        "n_fit_points": len(selected),
    }


def permutation_entropy(
    values: list[float],
    order: int,
    delay: int,
) -> dict[str, Any]:
    """Entropia de permutacion normalizada."""
    if order < 2 or delay < 1:
        raise ValueError("order>=2 y delay>=1")
    n_patterns = len(values) - (order - 1) * delay
    if n_patterns <= 0:
        raise ValueError("Serie demasiado corta para order/delay")
    counts: Counter[tuple[int, ...]] = Counter()
    for start in range(n_patterns):
        window = [(values[start + offset * delay], offset) for offset in range(order)]
        pattern = tuple(offset for _, offset in sorted(window, key=lambda item: (item[0], item[1])))
        counts[pattern] += 1
    entropy = 0.0
    for count in counts.values():
        probability = count / n_patterns
        entropy -= probability * math.log(probability)
    max_entropy = math.log(math.factorial(order))
    return {
        "order": order,
        "delay": delay,
        "permutation_entropy": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy else float("nan"),
        "n_patterns": n_patterns,
        "unique_patterns": len(counts),
        "max_patterns": math.factorial(order),
    }


def euclidean_distance_sq(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def linear_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    if len(x) != len(y) or len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    sxx = sum((value - x_mean) ** 2 for value in x)
    if sxx <= 0.0:
        return float("nan"), float("nan"), float("nan")
    sxy = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * value for value in x]
    ss_res = sum((y_value - fitted_value) ** 2 for y_value, fitted_value in zip(y, fitted))
    ss_tot = sum((y_value - y_mean) ** 2 for y_value in y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return slope, intercept, r2


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    if probability <= 0.0:
        return sorted_values[0]
    if probability >= 1.0:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
