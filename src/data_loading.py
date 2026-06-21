"""Funciones de ayuda para facilitar la entrada/salida para los CSV del proyecto."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


BASE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
    "taker_buy_quote",
]

FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "taker_buy_quote"]
INTEGER_COLUMNS = ["trades"]


def read_klines_csv(path: Path) -> list[dict[str, Any]]:
    """Lee el CSV limpio de klines y convierte columnas numericas."""
    rows: list[dict[str, Any]] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != BASE_COLUMNS:
            raise ValueError(
                "Columnas inesperadas en el CSV. "
                f"Esperadas={BASE_COLUMNS}, observadas={reader.fieldnames}"
            )

        for row in reader:
            parsed: dict[str, Any] = {"open_time": row["open_time"]}
            for column in FLOAT_COLUMNS:
                parsed[column] = float(row[column])
            for column in INTEGER_COLUMNS:
                parsed[column] = int(row[column])
            rows.append(parsed)

    return rows


def format_csv_value(value: Any) -> Any:
    """Formatea floats de forma compacta y estable para escritura CSV. 
       Usamos 12 dígitos significativos para evitar problemas de precisión al escribir y leer."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def write_rows_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Escribe filas de diccionarios a CSV con el orden de columnas indicado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_csv_value(row.get(column)) for column in columns})
