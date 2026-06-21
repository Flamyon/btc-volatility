"""Correlogramas y espectro sin dependencias externas."""

from __future__ import annotations

import math


def autocorrelation_fft(values: list[float], max_lag: int) -> list[float]:
    """Calcula ACF hasta max_lag usando convolucion por FFT."""
    if max_lag < 0:
        raise ValueError("max_lag no puede ser negativo")
    n = len(values)
    if n == 0:
        raise ValueError("Serie vacia")
    mean = sum(values) / n
    centered = [complex(value - mean, 0.0) for value in values]
    nfft = next_power_of_two(2 * n - 1)
    centered.extend([0j] * (nfft - n))
    fft_inplace(centered, inverse=False)
    for index, value in enumerate(centered):
        centered[index] = value * value.conjugate()
    fft_inplace(centered, inverse=True)
    denominator = centered[0].real
    if denominator <= 0:
        return [1.0] + [0.0] * max_lag
    return [centered[lag].real / denominator for lag in range(max_lag + 1)]


def pacf_levinson(acf: list[float], max_lag: int) -> list[float]:
    """PACF mediante recursiones de Levinson-Durbin."""
    if max_lag >= len(acf):
        raise ValueError("La ACF debe cubrir todos los retardos solicitados")
    pacf = [1.0]
    phi: list[float] = []
    prediction_error = acf[0]
    if prediction_error <= 0:
        return [1.0] + [0.0] * max_lag

    for lag in range(1, max_lag + 1):
        if lag == 1:
            reflection = acf[1] / prediction_error
            phi = [reflection]
        else:
            numerator = acf[lag] - sum(phi[j - 1] * acf[lag - j] for j in range(1, lag))
            reflection = numerator / prediction_error if prediction_error != 0 else 0.0
            updated = [
                phi[j] - reflection * phi[lag - 2 - j]
                for j in range(lag - 1)
            ]
            updated.append(reflection)
            phi = updated
        pacf.append(reflection)
        prediction_error *= max(1e-12, 1.0 - reflection * reflection)

    return pacf


def periodogram_fft(values: list[float]) -> list[dict[str, float]]:
    """Periodograma tras restar la media, con periodo equivalente en retardos."""
    n = len(values)
    if n == 0:
        raise ValueError("Serie vacia")
    mean = sum(values) / n
    nfft = next_power_of_two(n)
    data = [complex(value - mean, 0.0) for value in values]
    data.extend([0j] * (nfft - n))
    fft_inplace(data, inverse=False)

    result: list[dict[str, float]] = []
    half = nfft // 2
    scale = 1.0 / n
    for k in range(1, half + 1):
        frequency = k / nfft
        period_lags = nfft / k
        power = (data[k].real * data[k].real + data[k].imag * data[k].imag) * scale
        result.append(
            {
                "frequency_cycles_per_observation": frequency,
                "period_lags": period_lags,
                "period_hours": period_lags * 5.0 / 60.0,
                "period_days": period_lags * 5.0 / 1440.0,
                "power": power,
            }
        )
    return result


def spectral_peaks(
    spectrum: list[dict[str, float]],
    top_n: int = 12,
    min_period_lags: float = 2.0,
    max_period_lags: float = 4032.0,
) -> list[dict[str, float]]:
    """
    Selecciona picos locales del periodograma en un rango de periodos.

    El rango por defecto cubre desde 10 minutos hasta 2 semanas, suficiente para
    inspeccionar las referencias de 1 hora, 1 dia y 1 semana sin dejar que los
    componentes de muy baja frecuencia dominen toda la tabla.
    """
    candidates: list[dict[str, float]] = []
    for index in range(1, len(spectrum) - 1):
        current = spectrum[index]
        period = current["period_lags"]
        if period < min_period_lags or period > max_period_lags:
            continue
        if (
            current["power"] >= spectrum[index - 1]["power"]
            and current["power"] >= spectrum[index + 1]["power"]
        ):
            candidates.append(current)

    candidates.sort(key=lambda row: row["power"], reverse=True)
    peaks = candidates[:top_n]
    for rank, row in enumerate(peaks, start=1):
        row["rank"] = rank
        row["nearest_reference"] = nearest_reference(row["period_lags"])
    return peaks


def power_at_reference_periods(
    spectrum: list[dict[str, float]],
    reference_periods: list[float],
) -> list[dict[str, float]]:
    """Obtiene la potencia del bin mas cercano a periodos de referencia."""
    rows: list[dict[str, float]] = []
    for period in reference_periods:
        nearest = min(spectrum, key=lambda row: abs(row["period_lags"] - period))
        rows.append(
            {
                "reference_period_lags": period,
                "reference_label": reference_label(period),
                "nearest_period_lags": nearest["period_lags"],
                "nearest_period_hours": nearest["period_hours"],
                "nearest_period_days": nearest["period_days"],
                "power": nearest["power"],
            }
        )
    return rows


def nearest_reference(period_lags: float) -> str:
    references = [(12.0, "1 hora"), (288.0, "1 dia"), (2016.0, "1 semana")]
    period, label = min(references, key=lambda item: abs(math.log(period_lags / item[0])))
    relative_error = abs(period_lags - period) / period
    if relative_error <= 0.10:
        return label
    return ""


def reference_label(period_lags: float) -> str:
    if abs(period_lags - 12.0) < 1e-9:
        return "1 hora"
    if abs(period_lags - 288.0) < 1e-9:
        return "1 dia"
    if abs(period_lags - 2016.0) < 1e-9:
        return "1 semana"
    return f"{period_lags:g} retardos"


def fft_inplace(values: list[complex], inverse: bool = False) -> None:
    """FFT radix-2 iterativa in-place."""
    n = len(values)
    if n & (n - 1):
        raise ValueError("La longitud de la FFT debe ser potencia de 2")

    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            values[i], values[j] = values[j], values[i]

    length = 2
    sign = 1.0 if inverse else -1.0
    while length <= n:
        angle = sign * 2.0 * math.pi / length
        wlen = complex(math.cos(angle), math.sin(angle))
        half = length // 2
        for start in range(0, n, length):
            w = 1.0 + 0.0j
            for offset in range(half):
                u = values[start + offset]
                v = values[start + offset + half] * w
                values[start + offset] = u + v
                values[start + offset + half] = u - v
                w *= wlen
        length <<= 1

    if inverse:
        inv_n = 1.0 / n
        for index, value in enumerate(values):
            values[index] = value * inv_n


def next_power_of_two(value: int) -> int:
    """Siguiente potencia de dos mayor o igual que value."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()
