"""Filtrado lineal AR para la serie principal."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ARModel:
    """Modelo AR(p) estimado por Yule-Walker sobre serie estandarizada."""

    order: int
    intercept: float
    coefficients: list[float]
    innovation_variance: float
    aic: float
    bic: float
    nobs: int


def standardize_train_apply(
    values: list[float],
    train_end_index_exclusive: int,
) -> tuple[list[float], float, float]:
    """Estandariza toda la serie con media y desviacion tipica del entrenamiento."""
    train_values = values[:train_end_index_exclusive]
    mean = sum(train_values) / len(train_values)
    variance = sum((value - mean) ** 2 for value in train_values) / (len(train_values) - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        raise ValueError("La desviacion tipica de entrenamiento es cero")
    return [(value - mean) / std for value in values], mean, std


def select_ar_yule_walker(
    standardized_train: list[float],
    max_order: int = 100,
) -> tuple[ARModel, list[ARModel]]:
    """Ajusta AR(1..max_order) por Yule-Walker y selecciona BIC minimo."""
    if max_order < 1:
        raise ValueError("max_order debe ser positivo")
    if len(standardized_train) <= max_order + 1:
        raise ValueError("Entrenamiento demasiado corto para max_order")

    acf = autocorrelation_direct(standardized_train, max_order)
    phi: list[float] = []
    error_variance = 1.0
    models: list[ARModel] = []
    nobs = len(standardized_train)

    for order in range(1, max_order + 1):
        if order == 1:
            reflection = acf[1]
            phi = [reflection]
        else:
            numerator = acf[order] - sum(
                phi[j - 1] * acf[order - j] for j in range(1, order)
            )
            reflection = numerator / error_variance if error_variance > 0.0 else 0.0
            updated = [
                phi[j] - reflection * phi[order - 2 - j]
                for j in range(order - 1)
            ]
            updated.append(reflection)
            phi = updated

        error_variance *= max(1e-12, 1.0 - reflection * reflection)
        params = order + 1
        sigma2 = max(error_variance, 1e-300)
        aic = nobs * math.log(sigma2) + 2.0 * params
        bic = nobs * math.log(sigma2) + math.log(nobs) * params
        models.append(
            ARModel(
                order=order,
                intercept=0.0,
                coefficients=phi[:],
                innovation_variance=sigma2,
                aic=aic,
                bic=bic,
                nobs=nobs,
            )
        )

    selected = min(models, key=lambda model: model.bic)
    return selected, models


def autocorrelation_direct(values: list[float], max_lag: int) -> list[float]:
    """ACF directa para retardos pequenos."""
    n = len(values)
    mean = sum(values) / n
    centered = [value - mean for value in values]
    denominator = sum(value * value for value in centered)
    if denominator <= 0.0:
        return [1.0] + [0.0] * max_lag
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        numerator = sum(centered[index] * centered[index - lag] for index in range(lag, n))
        acf.append(numerator / denominator)
    return acf


def ar_residuals(
    standardized_values: list[float],
    model: ARModel,
) -> tuple[list[float | None], list[float | None]]:
    """Calcula fitted values y residuos one-step del AR seleccionado."""
    p = model.order
    fitted: list[float | None] = [None] * len(standardized_values)
    residuals: list[float | None] = [None] * len(standardized_values)
    coeffs = model.coefficients
    intercept = model.intercept

    for index in range(p, len(standardized_values)):
        prediction = intercept
        for lag in range(1, p + 1):
            prediction += coeffs[lag - 1] * standardized_values[index - lag]
        fitted[index] = prediction
        residuals[index] = standardized_values[index] - prediction

    return fitted, residuals


def ljung_box(acf: list[float], nobs: int, lags: list[int], fitted_order: int = 0) -> list[dict[str, float | int | str]]:
    """Ljung-Box Q para varios retardos usando ACF de residuos."""
    rows: list[dict[str, float | int | str]] = []
    for lag in lags:
        q_stat = nobs * (nobs + 2.0) * sum(
            (acf[k] * acf[k]) / (nobs - k) for k in range(1, lag + 1)
        )
        df = lag - fitted_order
        if df > 0:
            p_value = chi_square_sf(q_stat, df)
            p_value_text: float | str = p_value
        else:
            p_value_text = ""
        rows.append(
            {
                "lag": lag,
                "q_stat": q_stat,
                "df_adjusted": df if df > 0 else "",
                "p_value": p_value_text,
                "reject_5pct": (p_value_text < 0.05) if isinstance(p_value_text, float) else "",
            }
        )
    return rows


def chi_square_sf(x: float, df: int) -> float:
    """Funcion de supervivencia chi-cuadrado via gamma incompleta regularizada."""
    if x < 0.0 or df <= 0:
        raise ValueError("Argumentos invalidos para chi_square_sf")
    return gammq(0.5 * df, 0.5 * x)


def gammq(a: float, x: float) -> float:
    """Gamma incompleta regularizada superior Q(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("Argumentos invalidos para gammq")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, 1.0 - _gser(a, x))
    return min(1.0, _gcf(a, x))


def _gser(a: float, x: float) -> float:
    eps = 3e-14
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(10_000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            return total * math.exp(-x + a * math.log(x) - gln)
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float) -> float:
    eps = 3e-14
    fpmin = 1e-300
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / max(b, fpmin)
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return math.exp(-x + a * math.log(x) - gln) * h
    return math.exp(-x + a * math.log(x) - gln) * h
