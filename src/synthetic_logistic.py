"""Generacion del mapa logistico para validacion metodologica."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticSeries:
    name: str
    values: list[float]
    noise_level: str
    noise_sigma: float
    clipped_count: int


def logistic_map_series(
    r: float,
    x0: float,
    n_total: int,
    burn_in: int,
) -> list[float]:
    """Genera el mapa logistico y descarta el transitorio inicial."""
    if not 0.0 < x0 < 1.0:
        raise ValueError("x0 debe estar en (0, 1)")
    if n_total <= burn_in + 2:
        raise ValueError("n_total debe ser mayor que burn_in")
    values: list[float] = []
    x = x0
    for _ in range(n_total):
        values.append(x)
        x = r * x * (1.0 - x)
    return values[burn_in:]


def noisy_observation(
    clean: list[float],
    sigma: float,
    seed: int,
) -> tuple[list[float], int]:
    """Anade ruido observacional gaussiano y recorta a [0, 1]."""
    rng = random.Random(seed)
    output: list[float] = []
    clipped = 0
    for value in clean:
        noisy = value + rng.gauss(0.0, sigma)
        clipped_value = min(1.0, max(0.0, noisy))
        if clipped_value != noisy:
            clipped += 1
        output.append(clipped_value)
    return output, clipped


def build_logistic_series(
    r: float = 4.0,
    x0: float = 0.123456789,
    n_total: int = 12000,
    burn_in: int = 1000,
    seed: int = 20260603,
) -> list[SyntheticSeries]:
    clean = logistic_map_series(r, x0, n_total, burn_in)
    std = sample_std(clean)
    small_sigma = 0.01 * std
    moderate_sigma = 0.05 * std
    small, small_clipped = noisy_observation(clean, small_sigma, seed + 1)
    moderate, moderate_clipped = noisy_observation(clean, moderate_sigma, seed + 2)
    return [
        SyntheticSeries("logistic_clean", clean, "none", 0.0, 0),
        SyntheticSeries("logistic_noise_small", small, "small", small_sigma, small_clipped),
        SyntheticSeries("logistic_noise_moderate", moderate, "moderate", moderate_sigma, moderate_clipped),
    ]


def sample_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
