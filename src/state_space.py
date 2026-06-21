"""Reconstruccion del espacio de estados por retardos."""

from __future__ import annotations

from bisect import bisect_right
import io
import json
import math
import random
import struct
import zipfile
from pathlib import Path
from typing import Any, Iterable


def standardize_train(
    values: list[float],
    train_end_index_exclusive: int,
) -> tuple[list[float], float, float]:
    """Estandariza toda la serie con media y desviacion tipica del train."""
    train = values[:train_end_index_exclusive]
    mean = sum(train) / len(train)
    variance = sum((value - mean) ** 2 for value in train) / (len(train) - 1)
    std = math.sqrt(variance)
    if std <= 0.0:
        raise ValueError("Desviacion tipica de entrenamiento no positiva")
    return [(value - mean) / std for value in values], mean, std


def quantile_edges(values: list[float], bins: int) -> list[float]:
    """Bordes internos por cuantiles para discretizacion robusta."""
    if bins < 2:
        raise ValueError("bins debe ser al menos 2")
    sorted_values = sorted(values)
    edges: list[float] = []
    n = len(sorted_values)
    for index in range(1, bins):
        position = (n - 1) * index / bins
        lower = int(position)
        upper = min(lower + 1, n - 1)
        weight = position - lower
        value = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
        edges.append(value)
    return edges


def discretize(values: list[float], edges: list[float]) -> list[int]:
    """Asigna cada valor a un bin usando bordes internos."""
    return [bisect_right(edges, value) for value in values]


def average_mutual_information_from_bins(
    binned_values: list[int],
    max_lag: int,
    bins: int,
) -> list[dict[str, float | int]]:
    """AMI entre x_t y x_{t+tau} para tau=1..max_lag."""
    n = len(binned_values)
    rows: list[dict[str, float | int]] = []
    for tau in range(1, max_lag + 1):
        pair_n = n - tau
        x_counts = [0] * bins
        y_counts = [0] * bins
        joint = [0] * (bins * bins)
        for index in range(pair_n):
            x_bin = binned_values[index]
            y_bin = binned_values[index + tau]
            x_counts[x_bin] += 1
            y_counts[y_bin] += 1
            joint[x_bin * bins + y_bin] += 1
        mi = 0.0
        inv_n = 1.0 / pair_n
        for x_bin in range(bins):
            px_count = x_counts[x_bin]
            if px_count == 0:
                continue
            for y_bin in range(bins):
                joint_count = joint[x_bin * bins + y_bin]
                if joint_count == 0:
                    continue
                py_count = y_counts[y_bin]
                mi += joint_count * inv_n * math.log(
                    (joint_count * pair_n) / (px_count * py_count)
                )
        rows.append({"tau": tau, "mutual_information": mi, "n_pairs": pair_n})
    return rows


def select_tau_from_ami(rows: list[dict[str, float | int]]) -> tuple[int, str]:
    """Selecciona tau por primer minimo local o, si no aparece, por codo."""
    values = [float(row["mutual_information"]) for row in rows]
    taus = [int(row["tau"]) for row in rows]
    for index in range(1, len(values) - 1):
        if values[index] < values[index - 1] and values[index] <= values[index + 1]:
            return taus[index], "primer minimo local de AMI"

    # Fallback: punto con maxima distancia a la recta que une extremos.
    x0, y0 = float(taus[0]), values[0]
    x1, y1 = float(taus[-1]), values[-1]
    denominator = math.hypot(y1 - y0, x1 - x0)
    if denominator <= 0.0:
        return taus[min(12, len(taus) - 1)], "fallback por serie AMI plana"
    best_index = 0
    best_distance = -1.0
    for index, (tau, value) in enumerate(zip(taus, values)):
        distance = abs((y1 - y0) * tau - (x1 - x0) * value + x1 * y0 - y1 * x0) / denominator
        if distance > best_distance:
            best_distance = distance
            best_index = index
    return taus[best_index], "fallback por codo de AMI"


def evenly_spaced_indices(start: int, end_exclusive: int, sample_size: int) -> list[int]:
    """Submuestra reproducible y ordenada de indices enteros."""
    n = end_exclusive - start
    if n <= 0:
        raise ValueError("Rango vacio para submuestreo")
    if n <= sample_size:
        return list(range(start, end_exclusive))
    if sample_size <= 1:
        return [start]
    return [
        start + round((n - 1) * index / (sample_size - 1))
        for index in range(sample_size)
    ]


def fnn_and_cao(
    values: list[float],
    train_end_index_exclusive: int,
    tau: int,
    max_dim: int,
    sample_size: int,
    theiler_window: int,
    rtol: float,
    atol: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Calcula falsos vecinos y metodo de Cao con una submuestra exacta."""
    valid_start = (max_dim + 1) * tau
    valid_end = train_end_index_exclusive
    if valid_start >= valid_end:
        raise ValueError("No hay observaciones suficientes para tau y max_dim")
    sample_times = evenly_spaced_indices(valid_start, valid_end, sample_size)
    series_std = sample_std(values[:train_end_index_exclusive])

    e_values: dict[int, float] = {}
    e_star_values: dict[int, float] = {}
    fnn_rows: list[dict[str, Any]] = []

    for dim in range(1, max_dim + 2):
        neighbors = nearest_neighbors(values, sample_times, dim, tau, theiler_window)
        ratios: list[float] = []
        star_diffs: list[float] = []
        false_count = 0
        fnn_available = dim <= max_dim

        for row_index, neighbor_index, distance_sq in neighbors:
            t_i = sample_times[row_index]
            t_j = sample_times[neighbor_index]
            extra_diff = values[t_i - dim * tau] - values[t_j - dim * tau]
            distance = math.sqrt(distance_sq)
            distance_next = math.sqrt(distance_sq + extra_diff * extra_diff)
            ratios.append(distance_next / max(distance, 1e-12))
            star_diffs.append(abs(extra_diff))
            if fnn_available:
                criterion_r = abs(extra_diff) / max(distance, 1e-12) > rtol
                criterion_a = distance_next / max(series_std, 1e-12) > atol
                if criterion_r or criterion_a:
                    false_count += 1

        e_values[dim] = sum(ratios) / len(ratios)
        e_star_values[dim] = sum(star_diffs) / len(star_diffs)
        if fnn_available:
            fraction = false_count / len(neighbors)
            fnn_rows.append(
                {
                    "m": dim,
                    "fnn_fraction": fraction,
                    "fnn_percent": 100.0 * fraction,
                    "n_used": len(neighbors),
                    "false_neighbors": false_count,
                    "theiler_window": theiler_window,
                    "rtol": rtol,
                    "atol": atol,
                }
            )

    cao_rows: list[dict[str, Any]] = []
    for dim in range(1, max_dim + 1):
        e1 = e_values[dim + 1] / e_values[dim] if e_values[dim] else float("nan")
        e2 = (
            e_star_values[dim + 1] / e_star_values[dim]
            if e_star_values[dim] > 0.0
            else float("nan")
        )
        cao_rows.append(
            {
                "m": dim,
                "E1": e1,
                "E2": e2,
                "E_m": e_values[dim],
                "E_star_m": e_star_values[dim],
                "n_used": len(sample_times),
            }
        )

    return fnn_rows, cao_rows, len(sample_times)


def nearest_neighbors(
    values: list[float],
    sample_times: list[int],
    dim: int,
    tau: int,
    theiler_window: int,
) -> list[tuple[int, int, float]]:
    """Vecino mas cercano exacto dentro de la submuestra."""
    n = len(sample_times)
    rows: list[tuple[int, int, float]] = []
    for row_index, t_i in enumerate(sample_times):
        best_index = -1
        best_distance = float("inf")
        for candidate_index, t_j in enumerate(sample_times):
            if row_index == candidate_index or abs(t_i - t_j) <= theiler_window:
                continue
            distance = 0.0
            for coord in range(dim):
                diff = values[t_i - coord * tau] - values[t_j - coord * tau]
                distance += diff * diff
                if distance >= best_distance:
                    break
            if distance < best_distance:
                best_distance = distance
                best_index = candidate_index
        if best_index < 0:
            raise ValueError("No se encontro vecino no trivial")
        rows.append((row_index, best_index, best_distance))
    return rows


def sample_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def select_m_from_fnn(rows: list[dict[str, Any]], threshold_percent: float) -> tuple[int, str]:
    """Primera dimension bajo umbral; si no, codo por mayor caida."""
    for row in rows:
        if float(row["fnn_percent"]) <= threshold_percent:
            return int(row["m"]), f"primer m con FNN <= {threshold_percent:g}%"
    best_row = min(rows, key=lambda row: float(row["fnn_percent"]))
    return int(best_row["m"]), "minimo porcentaje FNN observado; no cae bajo el umbral"


def select_m_from_cao(rows: list[dict[str, Any]], tolerance: float = 0.02) -> tuple[int, str]:
    """Dimension donde E1 se estabiliza de forma sencilla."""
    e1 = [float(row["E1"]) for row in rows]
    dims = [int(row["m"]) for row in rows]
    for index in range(2, len(e1) - 1):
        if abs(e1[index] - e1[index - 1]) <= tolerance and abs(e1[index + 1] - e1[index]) <= tolerance:
            return dims[index], f"estabilizacion local de E1 con tolerancia {tolerance:g}"
    if len(e1) <= 2:
        return dims[-1], "serie E1 demasiado corta"
    drops = [abs(e1[index] - e1[index - 1]) for index in range(1, len(e1))]
    best_index = min(range(1, len(drops)), key=lambda index: drops[index])
    return dims[best_index + 1], "cambio minimo de E1 observado; estabilizacion ambigua"


def select_m_from_cao_unity_plateau(
    rows: list[dict[str, Any]],
    tolerance: float = 0.10,
    min_tail: int = 3,
) -> tuple[int, str]:
    """Primera dimension desde la que E1 permanece cerca de su limite unitario."""
    e1 = [float(row["E1"]) for row in rows]
    dims = [int(row["m"]) for row in rows]
    for index, value in enumerate(e1):
        tail = e1[index:]
        if len(tail) >= min_tail and abs(value - 1.0) <= tolerance:
            if all(abs(candidate - 1.0) <= tolerance for candidate in tail):
                return (
                    dims[index],
                    f"inicio de meseta de E1 dentro de +/-{tolerance:g} respecto a 1",
                )
    return select_m_from_cao(rows)


def select_final_m(m_fnn: int, m_cao: int, max_dim: int) -> tuple[int, str]:
    """Combina FNN y Cao de forma conservadora para fases posteriores."""
    if abs(m_fnn - m_cao) <= 2:
        return max(2, min(max_dim, m_fnn)), "FNN y Cao son compatibles; se mantiene FNN como dimension operativa"
    return max(2, min(max_dim, m_fnn)), "FNN y Cao discrepan; se prioriza FNN y se deja Cao como cautela"


def build_embedding_rows(
    values: list[float],
    times: list[str],
    indices: list[int],
    tau: int,
    dim: int,
) -> tuple[list[list[float]], list[int], list[str]]:
    """Construye X_t=[z_t,z_{t-tau},...,z_{t-(m-1)tau}]."""
    vectors: list[list[float]] = []
    vector_indices: list[int] = []
    vector_times: list[str] = []
    min_index = (dim - 1) * tau
    for index in indices:
        if index < min_index:
            continue
        vectors.append([values[index - coord * tau] for coord in range(dim)])
        vector_indices.append(index)
        vector_times.append(times[index])
    return vectors, vector_indices, vector_times


def write_npz_embedding(
    path: Path,
    vectors: list[list[float]],
    indices: list[int],
    metadata: dict[str, Any],
) -> None:
    """Escribe un NPZ compatible con NumPy usando solo libreria estandar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_npy_float64_2d(archive, "X.npy", vectors)
        _write_npy_int64_1d(archive, "indices.npy", indices)
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=True))


def _write_npy_float64_2d(archive: zipfile.ZipFile, name: str, data: list[list[float]]) -> None:
    rows = len(data)
    cols = len(data[0]) if rows else 0
    with archive.open(name, "w") as handle:
        handle.write(npy_header("<f8", (rows, cols)))
        if rows and cols:
            fmt = "<" + "d" * cols
            for row in data:
                handle.write(struct.pack(fmt, *row))


def _write_npy_int64_1d(archive: zipfile.ZipFile, name: str, data: list[int]) -> None:
    with archive.open(name, "w") as handle:
        handle.write(npy_header("<i8", (len(data),)))
        chunk_size = 4096
        for start in range(0, len(data), chunk_size):
            chunk = data[start : start + chunk_size]
            handle.write(struct.pack("<" + "q" * len(chunk), *chunk))


def npy_header(descr: str, shape: tuple[int, ...]) -> bytes:
    """Cabecera NPY v1.0."""
    shape_text = "(" + ", ".join(str(item) for item in shape)
    if len(shape) == 1:
        shape_text += ","
    shape_text += ")"
    header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape_text + ", }"
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes


def shuffled_copy(values: list[int], seed: int) -> list[int]:
    """Copia barajada reproducible."""
    rng = random.Random(seed)
    shuffled = values[:]
    rng.shuffle(shuffled)
    return shuffled


def rows_around_tau(rows: list[dict[str, Any]], tau: int, radius: int = 5) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if abs(int(row["tau"]) - tau) <= radius
    ]


def sample_embedding_rows(
    vectors: list[list[float]],
    indices: list[int],
    times: list[str],
    split: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Muestra pequena del embedding para inspeccion CSV."""
    selected = evenly_spaced_indices(0, len(vectors), min(limit, len(vectors)))
    rows: list[dict[str, Any]] = []
    for output_index in selected:
        row = {
            "split": split,
            "open_time": times[output_index],
            "index": indices[output_index],
        }
        for coord_index, value in enumerate(vectors[output_index], start=1):
            row[f"x{coord_index}"] = value
        rows.append(row)
    return rows


def bytes_preview(path: Path, max_bytes: int = 100) -> bytes:
    """Pequena utilidad para pruebas manuales."""
    with path.open("rb") as handle:
        return handle.read(max_bytes)
