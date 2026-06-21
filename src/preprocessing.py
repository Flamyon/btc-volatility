"""Construccion de variables para la Fase 1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from data_loading import BASE_COLUMNS
from volatility import rolling_future_sum, rolling_past_sum, safe_log


CREATED_COLUMNS = [
    "log_close",
    "r",
    "r2",
    "abs_r",
    "hl_range",
    "log_volume",
    "log_trades",
    "rv_past_12",
    "rv_past_48",
    "rv_past_288",
    "rv_future_12",
    "rv_future_48",
    "log_rv_past_12",
    "log_rv_past_48",
    "log_rv_past_288",
    "log_rv_future_12",
    "log_rv_future_48",
]

FINAL_COLUMNS = BASE_COLUMNS + CREATED_COLUMNS

REQUIRED_AFTER_DROPNA = CREATED_COLUMNS

FEATURE_DEFINITIONS = {
    "log_close": "log(close)",
    "r": "log(close_t) - log(close_{t-1})",
    "r2": "r_t^2",
    "abs_r": "|r_t|",
    "hl_range": "log(high / low)",
    "log_volume": "log(volume + 1)",
    "log_trades": "log(trades + 1)",
    "rv_past_12": "sum_{i=0}^{11} r_{t-i}^2",
    "rv_past_48": "sum_{i=0}^{47} r_{t-i}^2",
    "rv_past_288": "sum_{i=0}^{287} r_{t-i}^2",
    "rv_future_12": "sum_{i=1}^{12} r_{t+i}^2",
    "rv_future_48": "sum_{i=1}^{48} r_{t+i}^2",
    "log_rv_past_12": "log(rv_past_12 + epsilon)",
    "log_rv_past_48": "log(rv_past_48 + epsilon)",
    "log_rv_past_288": "log(rv_past_288 + epsilon)",
    "log_rv_future_12": "log(rv_future_12 + epsilon)",
    "log_rv_future_48": "log(rv_future_48 + epsilon)",
}


@dataclass(frozen=True)
class Phase1Result:
    """Resultado de la construccion de variables."""

    rows: list[dict[str, Any]]
    input_rows: int
    output_rows: int
    first_valid_index: int
    last_valid_index: int
    rows_dropped_start: int
    rows_dropped_end: int
    epsilon: float
    zero_rv_counts: dict[str, int]


def build_phase1_features(
    rows: list[dict[str, Any]],
    epsilon: float = 1e-12,
) -> Phase1Result:
    """Construye retornos, variables auxiliares y volatilidades realizadas."""
    if not rows:
        raise ValueError("No hay filas de entrada")

    log_close = [math.log(row["close"]) for row in rows]
    returns: list[float | None] = [None]
    for index in range(1, len(rows)):
        returns.append(log_close[index] - log_close[index - 1])

    squared_returns = [None if value is None else value * value for value in returns]

    rv_past_12 = rolling_past_sum(squared_returns, 12)
    rv_past_48 = rolling_past_sum(squared_returns, 48)
    rv_past_288 = rolling_past_sum(squared_returns, 288)
    rv_future_12 = rolling_future_sum(squared_returns, 12)
    rv_future_48 = rolling_future_sum(squared_returns, 48)

    zero_rv_counts = {
        "rv_past_12": _count_zero_or_negative(rv_past_12),
        "rv_past_48": _count_zero_or_negative(rv_past_48),
        "rv_past_288": _count_zero_or_negative(rv_past_288),
        "rv_future_12": _count_zero_or_negative(rv_future_12),
        "rv_future_48": _count_zero_or_negative(rv_future_48),
    }

    enriched_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        enriched = dict(row)
        r = returns[index]
        r2 = squared_returns[index]

        enriched.update(
            {
                "log_close": log_close[index],
                "r": r,
                "r2": r2,
                "abs_r": None if r is None else abs(r),
                "hl_range": math.log(row["high"] / row["low"]),
                "log_volume": math.log(row["volume"] + 1.0),
                "log_trades": math.log(row["trades"] + 1.0),
                "rv_past_12": rv_past_12[index],
                "rv_past_48": rv_past_48[index],
                "rv_past_288": rv_past_288[index],
                "rv_future_12": rv_future_12[index],
                "rv_future_48": rv_future_48[index],
                "log_rv_past_12": _log_if_present(rv_past_12[index], epsilon),
                "log_rv_past_48": _log_if_present(rv_past_48[index], epsilon),
                "log_rv_past_288": _log_if_present(rv_past_288[index], epsilon),
                "log_rv_future_12": _log_if_present(rv_future_12[index], epsilon),
                "log_rv_future_48": _log_if_present(rv_future_48[index], epsilon),
            }
        )
        enriched_rows.append(enriched)

    final_rows = [
        row
        for row in enriched_rows
        if all(row[column] is not None for column in REQUIRED_AFTER_DROPNA)
    ]

    if not final_rows:
        raise ValueError("Todas las filas fueron eliminadas tras construir ventanas")

    first_open_time = final_rows[0]["open_time"]
    last_open_time = final_rows[-1]["open_time"]
    first_valid_index = next(
        index for index, row in enumerate(enriched_rows) if row["open_time"] == first_open_time
    )
    last_valid_index = next(
        index for index, row in enumerate(enriched_rows) if row["open_time"] == last_open_time
    )

    return Phase1Result(
        rows=final_rows,
        input_rows=len(rows),
        output_rows=len(final_rows),
        first_valid_index=first_valid_index,
        last_valid_index=last_valid_index,
        rows_dropped_start=first_valid_index,
        rows_dropped_end=len(rows) - last_valid_index - 1,
        epsilon=epsilon,
        zero_rv_counts=zero_rv_counts,
    )


def _log_if_present(value: float | None, epsilon: float) -> float | None:
    if value is None:
        return None
    return safe_log(value, epsilon=epsilon)


def _count_zero_or_negative(values: list[float | None]) -> int:
    return sum(1 for value in values if value is not None and value <= 0.0)
