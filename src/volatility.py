"""Funciones para calcular retornos y volatilidad realizada partiendo de klines."""

from __future__ import annotations

import math


def safe_log(value: float, epsilon: float = 1e-12) -> float:
    """Logaritmo protegido frente a ceros (nunca devuelve -inf log(1e-12) ~ -27.63)."""
    return math.log(value + epsilon)


def rolling_past_sum(values: list[float | None], window: int) -> list[float | None]:
    """
    Suma movil pasada inclusiva.

    Para cada instante t devuelve sum(values[t-window+1:t+1]) solo si la ventana
    completa existe y todos sus valores son validos.
    """
    return _rolling_range_sum(values, window=window, offset_start=-(window - 1), offset_end=1)


def rolling_future_sum(values: list[float | None], window: int) -> list[float | None]:
    """
    Suma movil futura estrictamente posterior.

    Para cada instante t devuelve sum(values[t+1:t+window+1]). Esta definicion
    evita incluir r_t dentro del target futuro.
    """
    return _rolling_range_sum(values, window=window, offset_start=1, offset_end=window + 1)


def _rolling_range_sum(
    values: list[float | None],
    window: int,
    offset_start: int,
    offset_end: int,
) -> list[float | None]:
    """Calcula sumas de rangos con prefijos y control de valores ausentes."""
    if window <= 0:
        raise ValueError("window debe ser positivo")

    prefix_sum = [0.0]
    prefix_count = [0]

    for value in values:
        if value is None:
            prefix_sum.append(prefix_sum[-1])
            prefix_count.append(prefix_count[-1])
        else:
            prefix_sum.append(prefix_sum[-1] + value)
            prefix_count.append(prefix_count[-1] + 1)

    result: list[float | None] = []
    n = len(values)

    for index in range(n):
        start = index + offset_start
        end = index + offset_end
        if start < 0 or end > n:
            result.append(None)
            continue

        observed_count = prefix_count[end] - prefix_count[start]
        if observed_count != window:
            result.append(None)
            continue

        result.append(prefix_sum[end] - prefix_sum[start])

    return result
