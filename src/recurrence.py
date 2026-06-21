"""Funciones para graficos de recurrencia iniciales."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import zlib


def zscore(values: list[float]) -> tuple[list[float], float, float]:
    """Normaliza una serie con z-score usando solo la ventana local."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0 for _ in values], mean, std
    return [(value - mean) / std for value in values], mean, std


def recurrence_epsilon_for_rr(values: list[float], target_rr: float) -> tuple[float, float]:
    """Busca epsilon para aproximar una tasa de recurrencia objetivo en 1D."""
    if not 0.0 < target_rr < 1.0:
        raise ValueError("target_rr debe estar entre 0 y 1")
    sorted_values = sorted(values)
    n = len(sorted_values)
    low = 0.0
    high = sorted_values[-1] - sorted_values[0]

    for _ in range(50):
        mid = (low + high) / 2.0
        rr = recurrence_rate_from_sorted(sorted_values, mid)
        if rr < target_rr:
            low = mid
        else:
            high = mid

    epsilon = high
    achieved_rr = recurrence_rate_from_sorted(sorted_values, epsilon)
    return epsilon, achieved_rr


def recurrence_rate_from_sorted(sorted_values: list[float], epsilon: float) -> float:
    """Cuenta pares recurrentes ordenados, incluyendo diagonal, para una serie ordenada."""
    n = len(sorted_values)
    unordered_pairs = 0
    right = 0
    for left, value in enumerate(sorted_values):
        if right < left + 1:
            right = left + 1
        while right < n and sorted_values[right] - value <= epsilon:
            right += 1
        unordered_pairs += right - left - 1
    recurrence_count = n + 2 * unordered_pairs
    return recurrence_count / (n * n)


def write_recurrence_png(
    path: Path,
    values: list[float],
    epsilon: float,
) -> float:
    """
    Escribe un recurrence plot binario como PNG gris.

    Pixel negro = par recurrente, pixel blanco = par no recurrente. Devuelve la
    tasa de recurrencia exacta de la imagen escrita.
    """
    n = len(values)
    raw = bytearray()
    recurrence_count = 0
    local_values = values
    local_epsilon = epsilon

    for row_value in local_values:
        raw.append(0)  # PNG filter type 0.
        row = bytearray(n)
        for col_index, col_value in enumerate(local_values):
            if abs(row_value - col_value) <= local_epsilon:
                row[col_index] = 0
                recurrence_count += 1
            else:
                row[col_index] = 255
        raw.extend(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_grayscale_png(path, width=n, height=n, raw_scanlines=bytes(raw))
    return recurrence_count / (n * n)


def write_grayscale_png(path: Path, width: int, height: int, raw_scanlines: bytes) -> None:
    """Escribe un PNG grayscale de 8 bits con libreria estandar."""
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        payload = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    compressed = zlib.compress(raw_scanlines, level=6)
    png = signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png)
