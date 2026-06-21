"""Modelo HAR-logRV compacto para volatilidad realizada."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


HAR_FEATURES = ["log_rv_past_12", "log_rv_past_48", "log_rv_past_288"]
HAR_TARGET = "log_rv_future_12"


@dataclass(frozen=True)
class HARLogRVModel:
    """OLS compacto con intercepto y tres escalas pasadas de logRV."""

    intercept: float
    coefficients: list[float]
    feature_names: list[str]
    target: str
    ridge_lambda_used: float
    train_n: int


def fit_har_logrv_ols(X: list[list[float]], y: list[float]) -> HARLogRVModel:
    """Ajusta HAR-logRV por ecuaciones normales.

    Si la matriz normal queda numericamente singular, se reintenta con una pequena
    regularizacion ridge (1e-8) solo como estabilizacion computacional.
    """
    if len(X) != len(y) or not X:
        raise ValueError("X/y invalidos para HAR-logRV")
    for row in X:
        if len(row) != len(HAR_FEATURES):
            raise ValueError("Cada fila X debe contener tres features HAR")

    design = [[1.0, *row] for row in X]
    xtx, xty = normal_equations(design, y)
    ridge = 0.0
    try:
        beta = solve_linear_system(xtx, xty)
    except ValueError:
        ridge = 1e-8
        beta = solve_linear_system(add_ridge(xtx, ridge), xty)

    return HARLogRVModel(
        intercept=beta[0],
        coefficients=beta[1:],
        feature_names=HAR_FEATURES[:],
        target=HAR_TARGET,
        ridge_lambda_used=ridge,
        train_n=len(y),
    )


def predict_har_logrv(model: HARLogRVModel | dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    """Predice `log_rv_future_12` para filas con las tres features HAR."""
    model_obj = model_from_dict(model) if isinstance(model, dict) else model
    predictions: list[float] = []
    for row in rows:
        features = har_feature_row(row)
        prediction = model_obj.intercept + sum(
            coefficient * value
            for coefficient, value in zip(model_obj.coefficients, features)
        )
        predictions.append(prediction)
    return predictions


def har_feature_row(row: dict[str, Any]) -> list[float]:
    """Extrae las tres features HAR en orden fijo."""
    values = [float(row[name]) for name in HAR_FEATURES]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Features HAR no finitas")
    return values


def save_har_artifact(
    path: Path,
    model: HARLogRVModel,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Guarda el artefacto HAR en JSON para consumo desde el MVP."""
    artifact = {
        "model_name": "har_logrv_compact",
        "version": "1.0",
        **metadata,
        "target": model.target,
        "features": model.feature_names,
        "intercept": model.intercept,
        "coefficients": {
            feature: coefficient
            for feature, coefficient in zip(model.feature_names, model.coefficients)
        },
        "coefficient_vector": model.coefficients,
        "train_n": model.train_n,
        "ridge_lambda_used": model.ridge_lambda_used,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(artifact), indent=2, ensure_ascii=True), encoding="utf-8")


def load_har_artifact(path: Path) -> HARLogRVModel:
    """Carga un artefacto HAR-logRV exportado."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    coefficients = artifact.get("coefficient_vector")
    if coefficients is None:
        coeff_map = artifact["coefficients"]
        coefficients = [float(coeff_map[feature]) for feature in artifact["features"]]
    return HARLogRVModel(
        intercept=float(artifact["intercept"]),
        coefficients=[float(value) for value in coefficients],
        feature_names=[str(value) for value in artifact["features"]],
        target=str(artifact["target"]),
        ridge_lambda_used=float(artifact.get("ridge_lambda_used", 0.0)),
        train_n=int(artifact.get("train_n", 0)),
    )


def model_from_dict(value: dict[str, Any]) -> HARLogRVModel:
    coefficients = value.get("coefficient_vector")
    if coefficients is None and isinstance(value.get("coefficients"), dict):
        coefficients = [value["coefficients"][feature] for feature in value["features"]]
    return HARLogRVModel(
        intercept=float(value["intercept"]),
        coefficients=[float(item) for item in coefficients],
        feature_names=[str(item) for item in value.get("features", HAR_FEATURES)],
        target=str(value.get("target", HAR_TARGET)),
        ridge_lambda_used=float(value.get("ridge_lambda_used", 0.0)),
        train_n=int(value.get("train_n", 0)),
    )


def normal_equations(design: list[list[float]], y: list[float]) -> tuple[list[list[float]], list[float]]:
    cols = len(design[0])
    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [0.0 for _ in range(cols)]
    for row, target in zip(design, y):
        for i in range(cols):
            xty[i] += row[i] * target
            for j in range(cols):
                xtx[i][j] += row[i] * row[j]
    return xtx, xty


def add_ridge(matrix: list[list[float]], ridge: float) -> list[list[float]]:
    adjusted = [row[:] for row in matrix]
    for index in range(len(adjusted)):
        adjusted[index][index] += ridge
    return adjusted


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Resuelve Ax=b con eliminacion gaussiana y pivoteo parcial."""
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        pivot = augmented[pivot_row][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matriz singular o mal condicionada")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[row][n] for row in range(n)]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "__dataclass_fields__"):
        return clean_json(asdict(value))
    return value
