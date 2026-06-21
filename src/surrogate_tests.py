"""Utilidades para contrastes con series barajadas y subrogadas."""

from __future__ import annotations

import math
import random
from typing import Any

from dynamics_quantification import (
    correlation_curve_from_distances,
    linear_fit,
    pairwise_distances_sorted,
    permutation_entropy,
    radii_from_distances,
    select_evenly_spaced_pairs,
    summarize_correlation_dimension,
    summarize_lyapunov,
)
from spectral import fft_inplace
from state_space import build_embedding_rows


STAT_COLUMNS = [
    "d2",
    "lyapunov_slope_per_step",
    "lyapunov_slope_per_hour",
    "permutation_entropy_delay_1",
    "permutation_entropy_delay_tau",
]


def phase_randomized(values: list[float], rng: random.Random) -> list[float]:
    """Subrogado phase-randomized para N potencia de dos."""
    n = len(values)
    if n & (n - 1):
        raise ValueError("phase_randomized requiere N potencia de dos")
    spectrum = [complex(value, 0.0) for value in values]
    fft_inplace(spectrum, inverse=False)
    half = n // 2
    for k in range(1, half):
        amplitude = abs(spectrum[k])
        phase = rng.uniform(0.0, 2.0 * math.pi)
        value = complex(amplitude * math.cos(phase), amplitude * math.sin(phase))
        spectrum[k] = value
        spectrum[n - k] = value.conjugate()
    spectrum[0] = complex(spectrum[0].real, 0.0)
    spectrum[half] = complex(spectrum[half].real, 0.0)
    fft_inplace(spectrum, inverse=True)
    return [value.real for value in spectrum]


def aaft_surrogate(values: list[float], rng: random.Random) -> list[float]:
    """Subrogado AAFT aproximado."""
    gaussianized = gaussianize_by_rank(values)
    phase_random = phase_randomized(gaussianized, rng)
    sorted_original = sorted(values)
    ranked_phase_indices = sorted(range(len(values)), key=lambda index: (phase_random[index], index))
    surrogate = [0.0] * len(values)
    for rank, index in enumerate(ranked_phase_indices):
        surrogate[index] = sorted_original[rank]
    return surrogate


def gaussianize_by_rank(values: list[float]) -> list[float]:
    """Transforma a normal estandar por rangos."""
    n = len(values)
    ranked_indices = sorted(range(n), key=lambda index: (values[index], index))
    output = [0.0] * n
    for rank, index in enumerate(ranked_indices):
        probability = (rank + 0.5) / n
        output[index] = inverse_normal_cdf(probability)
    return output


def inverse_normal_cdf(p: float) -> float:
    """Aproximacion de Acklam para la normal inversa."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p debe estar en (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def compute_surrogate_statistics(
    values: list[float],
    tau: int,
    dim: int,
    theiler_window: int,
    corr_sample_size: int,
    corr_radii_count: int,
    lyap_block_size: int,
    lyap_reference_count: int,
    lyap_k_max: int,
    lyap_fit_start: int,
    lyap_fit_end: int,
) -> dict[str, float]:
    """Calcula los estadisticos de fase 10 para una ventana escalar."""
    times = [str(index) for index in range(len(values))]
    indices = list(range(len(values)))
    vectors, vector_indices, _ = build_embedding_rows(values, times, indices, tau, dim)

    d2 = float("nan")
    try:
        sample_vectors, sample_indices = select_evenly_spaced_pairs(
            vectors,
            vector_indices,
            corr_sample_size,
        )
        distances = pairwise_distances_sorted(sample_vectors, sample_indices, theiler_window)
        radii = radii_from_distances(distances, corr_radii_count)
        corr_rows = correlation_curve_from_distances(
            distances,
            radii,
            "series",
            len(sample_vectors),
            theiler_window,
        )
        corr_summary = summarize_correlation_dimension(corr_rows)
        if corr_summary["d2_estimate"] is not None:
            d2 = float(corr_summary["d2_estimate"])
    except Exception:
        d2 = float("nan")

    lyap_step = float("nan")
    lyap_hour = float("nan")
    try:
        block_start = max(0, len(vectors) // 2 - lyap_block_size // 2)
        block = vectors[block_start : block_start + lyap_block_size]
        lyap_rows, _ = rosenstein_curve_sampled(
            block,
            theiler_window=theiler_window,
            k_max=lyap_k_max,
            reference_count=lyap_reference_count,
            series_label="series",
        )
        lyap_summary = summarize_lyapunov(lyap_rows, lyap_fit_start, lyap_fit_end)
        lyap_step = float(lyap_summary["slope_per_5min_step"])
        lyap_hour = float(lyap_summary["slope_per_hour"])
    except Exception:
        lyap_step = float("nan")
        lyap_hour = float("nan")

    pe_delay_1 = float("nan")
    pe_delay_tau = float("nan")
    try:
        pe_delay_1 = float(permutation_entropy(values, 5, 1)["normalized_entropy"])
    except Exception:
        pe_delay_1 = float("nan")
    try:
        pe_delay_tau = float(permutation_entropy(values, 5, tau)["normalized_entropy"])
    except Exception:
        pe_delay_tau = float("nan")

    return {
        "d2": d2,
        "lyapunov_slope_per_step": lyap_step,
        "lyapunov_slope_per_hour": lyap_hour,
        "permutation_entropy_delay_1": pe_delay_1,
        "permutation_entropy_delay_tau": pe_delay_tau,
    }


def rosenstein_curve_sampled(
    vectors: list[list[float]],
    theiler_window: int,
    k_max: int,
    reference_count: int,
    series_label: str,
) -> tuple[list[dict[str, Any]], int]:
    """Rosenstein aproximado usando un subconjunto de puntos de referencia."""
    valid_n = len(vectors) - k_max
    if valid_n <= theiler_window + 2:
        raise ValueError("Bloque demasiado corto para k_max y Theiler")
    references = evenly_spaced_indices(0, valid_n, min(reference_count, valid_n))
    neighbors: list[tuple[int, int]] = []
    for i in references:
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
    if not neighbors:
        raise ValueError("No hay vecinos validos para Rosenstein")

    rows: list[dict[str, Any]] = []
    for k in range(k_max + 1):
        log_distances: list[float] = []
        for i, j in neighbors:
            distance = math.sqrt(euclidean_distance_sq(vectors[i + k], vectors[j + k]))
            if distance > 1e-12:
                log_distances.append(math.log(distance))
        rows.append(
            {
                "series": series_label,
                "k": k,
                "time_minutes": 5 * k,
                "mean_log_distance": sum(log_distances) / len(log_distances),
                "n_pairs": len(log_distances),
                "theiler_window": theiler_window,
            }
        )
    return rows, len(neighbors)


def summarize_surrogate_group(
    group: str,
    original_stats: dict[str, float],
    replicate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resumen por estadistico frente a un grupo de subrogadas."""
    rows: list[dict[str, Any]] = []
    for stat in STAT_COLUMNS:
        values = [
            float(row[stat])
            for row in replicate_rows
            if isinstance(row.get(stat), (int, float)) and math.isfinite(float(row[stat]))
        ]
        original = float(original_stats.get(stat, float("nan")))
        if not values:
            rows.append(
                {
                    "group": group,
                    "statistic": stat,
                    "original": original,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "p05": float("nan"),
                    "median": float("nan"),
                    "p95": float("nan"),
                    "max": float("nan"),
                    "S": float("nan"),
                    "empirical_p_value": float("nan"),
                    "n_success": 0,
                    "n_failures": len(replicate_rows),
                }
            )
            continue
        ordered = sorted(values)
        mean = sum(values) / len(values)
        std = sample_std(values)
        if std > 0.0 and math.isfinite(original):
            s_value = abs(original - mean) / std
            distance_original = abs(original - mean)
            count_extreme = sum(1 for value in values if abs(value - mean) >= distance_original)
            p_value = (1 + count_extreme) / (len(values) + 1)
        else:
            s_value = float("nan")
            p_value = float("nan")
        rows.append(
            {
                "group": group,
                "statistic": stat,
                "original": original,
                "mean": mean,
                "std": std,
                "min": ordered[0],
                "p05": quantile(ordered, 0.05),
                "median": quantile(ordered, 0.50),
                "p95": quantile(ordered, 0.95),
                "max": ordered[-1],
                "S": s_value,
                "empirical_p_value": p_value,
                "n_success": len(values),
                "n_failures": len(replicate_rows) - len(values),
            }
        )
    return rows


def evenly_spaced_indices(start: int, end_exclusive: int, sample_size: int) -> list[int]:
    n = end_exclusive - start
    if n <= sample_size:
        return list(range(start, end_exclusive))
    return [start + round((n - 1) * index / (sample_size - 1)) for index in range(sample_size)]


def euclidean_distance_sq(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def quantile(sorted_values: list[float], probability: float) -> float:
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
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))

