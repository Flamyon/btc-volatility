"""Herramientas de estacionariedad sin dependencias externas."""

from __future__ import annotations

from dataclasses import dataclass
import math


ADF_CRITICAL_VALUES_C = {
    "1%": -3.43,
    "5%": -2.86,
    "10%": -2.57,
}

KPSS_CRITICAL_VALUES_C = {
    "1%": 0.739,
    "5%": 0.463,
    "10%": 0.347,
}


@dataclass(frozen=True)
class ADFResult:
    """Resultado de un ADF con constante."""

    lag: int
    nobs: int
    statistic: float
    gamma: float
    std_error: float
    rss: float
    sigma2: float
    aic: float
    bic: float
    critical_1pct: float
    critical_5pct: float
    critical_10pct: float
    p_value_range: str
    reject_5pct: bool


@dataclass(frozen=True)
class KPSSResult:
    """Resultado de KPSS de estacionariedad en nivel."""

    nlags: int
    nobs: int
    statistic: float
    long_run_variance: float
    critical_1pct: float
    critical_5pct: float
    critical_10pct: float
    p_value_range: str
    reject_5pct: bool


def rolling_mean_std(
    values: list[float],
    window: int,
) -> tuple[list[float], list[float]]:
    """Calcula media y desviacion tipica rolling con ventana fija."""
    if window <= 1:
        raise ValueError("window debe ser mayor que 1")
    if len(values) < window:
        raise ValueError("La serie es mas corta que la ventana rolling")

    prefix_sum = [0.0]
    prefix_sum2 = [0.0]
    for value in values:
        prefix_sum.append(prefix_sum[-1] + value)
        prefix_sum2.append(prefix_sum2[-1] + value * value)

    means: list[float] = []
    stds: list[float] = []
    for end in range(window, len(values) + 1):
        start = end - window
        total = prefix_sum[end] - prefix_sum[start]
        total2 = prefix_sum2[end] - prefix_sum2[start]
        mean = total / window
        variance_num = total2 - total * total / window
        variance = max(0.0, variance_num / (window - 1))
        means.append(mean)
        stds.append(math.sqrt(variance))

    return means, stds


def adf_select_lag(
    values: list[float],
    candidate_lags: list[int] | None = None,
) -> tuple[ADFResult, list[ADFResult]]:
    """Ejecuta ADF con constante y selecciona lag por BIC."""
    if candidate_lags is None:
        candidate_lags = [0, 1, 2, 3, 6, 12]

    standardized = _standardize(values)
    results = [adf_test(standardized, lag=lag) for lag in candidate_lags]
    selected = min(results, key=lambda result: result.bic)
    return selected, results


def adf_test(values: list[float], lag: int) -> ADFResult:
    """
    ADF con constante:

    Delta y_t = alpha + gamma y_{t-1} + phi_1 Delta y_{t-1}
                + ... + phi_p Delta y_{t-p} + e_t.
    """
    if lag < 0:
        raise ValueError("lag no puede ser negativo")
    if len(values) <= lag + 2:
        raise ValueError("Serie demasiado corta para el lag solicitado")

    y = values
    dy = [0.0] + [y[index] - y[index - 1] for index in range(1, len(y))]
    k = 2 + lag
    xtx = [[0.0 for _ in range(k)] for _ in range(k)]
    xty = [0.0 for _ in range(k)]
    yty = 0.0
    nobs = 0

    for index in range(lag + 1, len(y)):
        target = dy[index]
        regressors = [1.0, y[index - 1]]
        for lag_index in range(1, lag + 1):
            regressors.append(dy[index - lag_index])

        nobs += 1
        yty += target * target
        for row in range(k):
            x_row = regressors[row]
            xty[row] += x_row * target
            for col in range(row, k):
                xtx[row][col] += x_row * regressors[col]

    for row in range(k):
        for col in range(row):
            xtx[row][col] = xtx[col][row]

    beta = solve_linear_system(xtx, xty)
    rss = max(1e-300, yty - sum(beta[index] * xty[index] for index in range(k)))
    dof = max(1, nobs - k)
    sigma2 = rss / dof

    unit = [0.0 for _ in range(k)]
    unit[1] = 1.0
    inv_column = solve_linear_system(xtx, unit)
    gamma_variance = max(0.0, sigma2 * inv_column[1])
    std_error = math.sqrt(gamma_variance) if gamma_variance > 0.0 else float("inf")
    statistic = beta[1] / std_error if std_error > 0.0 else float("-inf")
    aic = nobs * math.log(rss / nobs) + 2 * k
    bic = nobs * math.log(rss / nobs) + k * math.log(nobs)

    return ADFResult(
        lag=lag,
        nobs=nobs,
        statistic=statistic,
        gamma=beta[1],
        std_error=std_error,
        rss=rss,
        sigma2=sigma2,
        aic=aic,
        bic=bic,
        critical_1pct=ADF_CRITICAL_VALUES_C["1%"],
        critical_5pct=ADF_CRITICAL_VALUES_C["5%"],
        critical_10pct=ADF_CRITICAL_VALUES_C["10%"],
        p_value_range=_adf_p_value_range(statistic),
        reject_5pct=statistic < ADF_CRITICAL_VALUES_C["5%"],
    )


def kpss_test(values: list[float], nlags: int | None = None) -> KPSSResult:
    """KPSS de estacionariedad en nivel con varianza Newey-West."""
    standardized = _standardize(values)
    n = len(standardized)
    if nlags is None:
        nlags = int(12 * (n / 100) ** 0.25)

    mean = sum(standardized) / n
    residuals = [value - mean for value in standardized]
    cumulative = []
    running = 0.0
    for residual in residuals:
        running += residual
        cumulative.append(running)

    eta = sum(value * value for value in cumulative) / (n * n)
    gamma0 = sum(residual * residual for residual in residuals) / n
    long_run_variance = gamma0
    for lag in range(1, nlags + 1):
        covariance = sum(
            residuals[index] * residuals[index - lag] for index in range(lag, n)
        ) / n
        weight = 1.0 - lag / (nlags + 1.0)
        long_run_variance += 2.0 * weight * covariance

    statistic = eta / long_run_variance if long_run_variance > 0.0 else float("inf")
    return KPSSResult(
        nlags=nlags,
        nobs=n,
        statistic=statistic,
        long_run_variance=long_run_variance,
        critical_1pct=KPSS_CRITICAL_VALUES_C["1%"],
        critical_5pct=KPSS_CRITICAL_VALUES_C["5%"],
        critical_10pct=KPSS_CRITICAL_VALUES_C["10%"],
        p_value_range=_kpss_p_value_range(statistic),
        reject_5pct=statistic > KPSS_CRITICAL_VALUES_C["5%"],
    )


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Resuelve Ax=b por eliminacion gaussiana con pivoteo parcial."""
    n = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-14:
            raise ValueError("Sistema singular o mal condicionado")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        pivot_value = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot_value

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]

    return [augmented[row][n] for row in range(n)]


def _standardize(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _adf_p_value_range(statistic: float) -> str:
    if statistic < ADF_CRITICAL_VALUES_C["1%"]:
        return "< 0.01"
    if statistic < ADF_CRITICAL_VALUES_C["5%"]:
        return "0.01 - 0.05"
    if statistic < ADF_CRITICAL_VALUES_C["10%"]:
        return "0.05 - 0.10"
    return "> 0.10"


def _kpss_p_value_range(statistic: float) -> str:
    if statistic > KPSS_CRITICAL_VALUES_C["1%"]:
        return "< 0.01"
    if statistic > KPSS_CRITICAL_VALUES_C["5%"]:
        return "0.01 - 0.05"
    if statistic > KPSS_CRITICAL_VALUES_C["10%"]:
        return "0.05 - 0.10"
    return "> 0.10"
