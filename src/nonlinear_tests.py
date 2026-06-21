"""Contrastes basicos de dependencia no lineal residual."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from linear_filtering import chi_square_sf
from spectral import autocorrelation_fft


@dataclass(frozen=True)
class BDSResult:
    """Resultado BDS para una dimension y epsilon."""

    embedding_dim: int
    epsilon: float
    nobs: int
    correlation_sum_1: float
    correlation_sum_m: float
    variance: float
    statistic: float
    p_value: float


def zscore(values: list[float]) -> list[float]:
    """Normaliza una lista a media 0 y desviacion tipica muestral 1."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def bds_test_window(
    values: list[float],
    embedding_dims: list[int],
    epsilons: list[float],
) -> list[BDSResult]:
    """
    BDS en una ventana usando distancia supremo.

    Implementacion por bitsets para evitar matrices Python NxN de objetos. La
    ventana se asume ya estandarizada.
    """
    if not values:
        raise ValueError("Serie vacia para BDS")
    max_dim = max(embedding_dims)
    if max_dim > len(values):
        raise ValueError("La dimension de embedding supera la longitud de la ventana")
    results: list[BDSResult] = []

    for epsilon in epsilons:
        row_bits, row_counts = _indicator_bitsets(values, epsilon)
        c1 = sum(row_counts) / (len(values) * len(values))
        variance_k = _bds_k(row_counts, len(values))
        for embedding_dim in embedding_dims:
            cm = _correlation_sum_embedding(row_bits, len(values), embedding_dim)
            variance = _bds_variance(c1, variance_k, embedding_dim)
            effective_n = len(values) - embedding_dim + 1
            if variance <= 0.0:
                statistic = float("nan")
                p_value = float("nan")
            else:
                statistic = math.sqrt(effective_n) * (cm - c1**embedding_dim) / math.sqrt(variance)
                p_value = normal_two_sided_pvalue(statistic)
            results.append(
                BDSResult(
                    embedding_dim=embedding_dim,
                    epsilon=epsilon,
                    nobs=effective_n,
                    correlation_sum_1=c1,
                    correlation_sum_m=cm,
                    variance=variance,
                    statistic=statistic,
                    p_value=p_value,
                )
            )

    results.sort(key=lambda item: (item.embedding_dim, item.epsilon))
    return results


def _indicator_bitsets(values: list[float], epsilon: float) -> tuple[list[int], list[int]]:
    """Vecindades 1D con bitsets usando una ventana ordenada deslizante."""
    n = len(values)
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    row_bits = [0 for _ in values]
    row_counts = [0 for _ in values]
    lower = 0
    upper = 0
    bits = 0

    for row_value, original_index in sorted_pairs:
        while lower < n and sorted_pairs[lower][0] < row_value - epsilon:
            bits &= ~(1 << sorted_pairs[lower][1])
            lower += 1
        while upper < n and sorted_pairs[upper][0] <= row_value + epsilon:
            bits |= 1 << sorted_pairs[upper][1]
            upper += 1
        row_bits[original_index] = bits
        row_counts[original_index] = bits.bit_count()
    return row_bits, row_counts


def _bds_k(row_counts: list[int], n: int) -> float:
    total = sum(row_counts)
    numerator = sum(count * count for count in row_counts) - 3 * total + 2 * n
    denominator = n * (n - 1) * (n - 2)
    return numerator / denominator if denominator > 0 else 0.0


def _correlation_sum_embedding(row_bits: list[int], n: int, embedding_dim: int) -> float:
    effective_n = n - embedding_dim + 1
    mask = (1 << effective_n) - 1
    total = 0
    for start in range(effective_n):
        bits = mask
        for offset in range(embedding_dim):
            bits &= row_bits[start + offset] >> offset
        total += bits.bit_count()
    return total / (effective_n * effective_n)


def _bds_variance(c1: float, k: float, embedding_dim: int) -> float:
    total = 0.0
    for j in range(1, embedding_dim):
        total += (k ** (embedding_dim - j)) * (c1 ** (2 * j))
    return 4.0 * (
        k**embedding_dim
        + 2.0 * total
        + ((embedding_dim - 1) ** 2) * (c1 ** (2 * embedding_dim))
        - (embedding_dim**2) * k * (c1 ** (2 * embedding_dim - 2))
    )


def normal_two_sided_pvalue(z: float) -> float:
    """p-value bilateral normal usando erfc."""
    if not math.isfinite(z):
        return float("nan")
    return math.erfc(abs(z) / math.sqrt(2.0))


def arch_lm_yule_walker(values: list[float], lags: list[int]) -> list[dict[str, float | int | bool]]:
    """
    ARCH LM aproximado via AR(q) sobre residuos cuadrados centrados.

    Para q grande evita una regresion OLS explicita con matriz de diseno enorme.
    Con muestras grandes, el R2 de Yule-Walker sobre e_t^2 es una aproximacion
    estable para detectar heterocedasticidad condicional.
    """
    squares = [value * value for value in values]
    mean_sq = sum(squares) / len(squares)
    centered = [value - mean_sq for value in squares]
    max_lag = max(lags)
    acf = autocorrelation_fft(centered, max_lag)
    rows: list[dict[str, float | int | bool]] = []

    for lag in lags:
        innovation_variance = _yw_innovation_variance(acf, lag)
        r2 = max(0.0, min(1.0, 1.0 - innovation_variance))
        n_eff = len(values) - lag
        lm_stat = n_eff * r2
        p_value = chi_square_sf(lm_stat, lag)
        rows.append(
            {
                "lag": lag,
                "nobs_effective": n_eff,
                "lm_stat": lm_stat,
                "r_squared_yw": r2,
                "p_value": p_value,
                "reject_5pct": p_value < 0.05,
            }
        )
    return rows


def _yw_innovation_variance(acf: list[float], order: int) -> float:
    if order <= 0:
        return 1.0
    error = 1.0
    phi: list[float] = []
    for lag in range(1, order + 1):
        if lag == 1:
            reflection = acf[1]
            phi = [reflection]
        else:
            numerator = acf[lag] - sum(phi[j - 1] * acf[lag - j] for j in range(1, lag))
            reflection = numerator / error if error > 0.0 else 0.0
            updated = [
                phi[j] - reflection * phi[lag - 2 - j]
                for j in range(lag - 1)
            ]
            updated.append(reflection)
            phi = updated
        error *= max(1e-12, 1.0 - reflection * reflection)
    return max(0.0, error)


def squared_autocorrelation(values: list[float], lag: int) -> float:
    """Autocorrelacion de valores cuadrados en un retardo."""
    squares = [value * value for value in values]
    n = len(squares)
    if lag <= 0 or lag >= n:
        raise ValueError("Lag invalido")
    mean = sum(squares) / n
    denominator = sum((value - mean) ** 2 for value in squares)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(
        (squares[index] - mean) * (squares[index - lag] - mean)
        for index in range(lag, n)
    )
    return numerator / denominator


def shuffle_squared_acf_comparison(
    values: list[float],
    lags: list[int],
    n_shuffles: int,
    rng: random.Random,
) -> list[dict[str, float | int | str]]:
    """Compara autocorrelaciones de cuadrados original vs permutaciones."""
    original = {lag: squared_autocorrelation(values, lag) for lag in lags}
    shuffled_stats = {lag: [] for lag in lags}
    work = values[:]
    for _ in range(n_shuffles):
        rng.shuffle(work)
        for lag in lags:
            shuffled_stats[lag].append(squared_autocorrelation(work, lag))

    rows: list[dict[str, float | int | str]] = []
    for lag in lags:
        shuffled = shuffled_stats[lag]
        mean = sum(shuffled) / len(shuffled)
        variance = (
            sum((value - mean) ** 2 for value in shuffled) / (len(shuffled) - 1)
            if len(shuffled) > 1
            else 0.0
        )
        std = math.sqrt(variance)
        percentile = sum(1 for value in shuffled if value <= original[lag]) / len(shuffled)
        rows.append(
            {
                "statistic": f"acf_squared_lag_{lag}",
                "lag": lag,
                "original_value": original[lag],
                "shuffle_mean": mean,
                "shuffle_std": std,
                "empirical_percentile_original": percentile,
                "n_shuffles": n_shuffles,
            }
        )
    return rows
