# btc-volatility

Repositorio técnico asociado al Trabajo de Fin de Grado sobre análisis de volatilidad realizada de Bitcoin y exploración de herramientas de dinámica no lineal en series temporales financieras.

El objetivo del repositorio es reproducir el procedimiento computacional usado en la memoria: descarga y limpieza de datos, construcción de variables de volatilidad realizada, análisis estadístico, reconstrucción del espacio de estados, contrastes de no linealidad, predicción local y comparación final con modelos de referencia.

## Estructura del repositorio

```text
btc-volatility/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── model_artifacts/
│   └── synthetic/
├── reports/
│   └── FASES.md
├── src/
├── btc_5m_clean.csv
├── README.md
├── REPRODUCIBILITY.md
├── DATA_MANIFEST.md
└── requirements.txt
```

## Fuente de datos

Los datos proceden de Binance Spot, par `BTCUSDT`, intervalo `5m`, mediante los ficheros públicos mensuales de klines disponibles en:

```text
https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/5m/
```

Periodo utilizado:

```text
2024-01-01 00:00:00 UTC -- 2026-04-30 23:55:00 UTC
```

El dataset limpio principal es:

```text
btc_5m_clean.csv
```

La validación inicial confirmó:

- frecuencia de 5 minutos;
- 245088 observaciones;
- ausencia de duplicados;
- ausencia de huecos temporales;
- ausencia de valores nulos;
- precios positivos.

Los detalles de descarga, ficheros usados y hashes se documentan en `DATA_MANIFEST.md`.

## Fases del estudio

El procedimiento técnico se organiza por fases:

- Fase 0: validación integral de datos.
- Fase 1: construcción de variables de volatilidad realizada.
- Fase 2: visualización temporal y estadísticos descriptivos.
- Fase 3: contrastes de estacionariedad.
- Fase 4: correlogramas y análisis espectral.
- Fase 5: gráficos de recurrencia.
- Fase 6: filtrado lineal AR(p).
- Fase 7: contrastes de no linealidad.
- Fase 8: reconstrucción del espacio de estados.
- Fase 9: cuantificación dinámica.
- Fase 10: contrastes con datos subrogados.
- Fase 11: predicción local kNN.
- Fase 12: sensibilidad de parámetros.
- Fase 13: validación metodológica en mapa logístico.
- Fase 14: modelo HAR-logRV exportable al MVP.

El detalle completo de cada fase se conserva en:

```text
reports/FASES.md
```

## Instalación

Se recomienda usar un entorno virtual de Python.

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

La ejecución completa del pipeline puede realizarse mediante los scripts de `src/`:

```bash
python src/build_btc_5m_clean.py
python src/validate.py
python src/build_phase1_features.py
python src/phase2_general_tools.py
python src/phase3_stationarity.py
python src/phase4_correlogram_spectrum.py
python src/phase5_initial_recurrence.py
python src/phase6_linear_filtering.py
python src/phase7_nonlinearity_tests.py
python src/phase8_state_space_reconstruction.py
python src/phase9_dynamics_quantification.py
python src/phase10_surrogate_tests.py
python src/phase11_local_state_space_prediction.py
python src/phase12_prediction_sensitivity.py
python src/phase13_logistic_map_validation.py
python src/phase14_har_logrv_mvp.py
```

## Resultados principales

El análisis usa como serie principal `log_rv_past_12`, derivada de la volatilidad realizada pasada sobre ventanas de 12 velas de 5 minutos.

Modelos finales considerados:

- AR(49) como referencia lineal;
- reconstrucción del espacio de estados con `tau=137` y `m=5`;
- predicción local kNN con `k=50` a `k=200`;
- HAR-logRV como modelo práctico compacto para el MVP.

El modelo HAR-logRV se exporta como artefacto en:

```text
data/model_artifacts/har_logrv_model.json
```

## Licencia y uso

Este repositorio se proporciona como material técnico de apoyo a una memoria académica. El código tiene finalidad reproducible y experimental. Los datos originales pertenecen a la fuente pública indicada y deben usarse respetando sus condiciones de distribución.