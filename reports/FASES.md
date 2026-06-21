# Explicacion de cada fase del proyecto

## FASE 0: Validación integral de datos

**Actividades:**
- Test de columnas esperadas
- Rango temporal 2024-01-01 a 2026-04-30
- Frecuencia 5 minutos verificada
- Duplicados: 0
- Huecos temporales: 0
- Valores nulos: 0
- Precios positivos

**Resultados guardados:**
- `btc_5m_clean.csv` (validado, 245,088 obs)

**Conclusión:**
- Integridad de 245,088 observaciones validada sin gaps ni nulos.

---

## FASE 1: Construcción variables volatilidad realizada

**Actividades:**
- log_close
- Retornos logarítmicos r_t
- r_t² (cuadrado)
- |r_t| (absoluto)
- Rango high-low logarítmico
- Log(volume + 1)
- Log(trades + 1)
- Volatilidad realizada pasada: rv_past_12, rv_past_48, rv_past_288 (ventanas)
- Volatilidad realizada futura: rv_future_12, rv_future_48
- Transformación logarítmica rv con epsilon=1e-12
- Eliminación 288 filas iniciales (sin histórico)
- Eliminación 48 filas finales (sin futuro)
- Validación no-leakage

**Resultados guardados:**
- Datos: btc_5m_features.csv (244,752 obs)

**Conclusión:**
- Variables de volatilidad realizadas construidas y validadas sin leakage.
---

## FASE 2: Visualización temporal y estadísticos

**Actividades:**
- Gráficos temporales: close, log_close, r, |r|, log_rv_past_12, log_rv_future_12, volume, trades
- Estadísticos descriptivos: mean, std, min, percentiles, max, skewness, kurtosis
- Detección eventos extremos: retorno mínimo/máximo, máxima volatilidad, volumen extremo
- Comentarios visuales de patrones

**Resultados guardados:**
- SVG: 7 gráficos de series temporales
- CSV: descriptive_statistics, extreme_events

**Conclusión:**
- Estadísticos descriptivos: Retornos con colas pesadas (curtosis 53.6) y clustering visible.
- Eventos extremos: Identificados, particularmente episodio 2025-10-10.
- Conclusión global: log_rv_past_12 confirmada como serie principal.

---

## FASE 3: Tests estacionariedad


**Actividades:**
- Test ADF en: close, log_close, r, |r|, log_rv_past_12
- Test KPSS en mismas series
- Selección lag automática ADF
- Momentos rolling (media, desv. tipica) ventana=288 velas
- Comparación primera vs segunda mitad muestra

**Resultados guardados:**
- SVG: 10 gráficos (series + rolling)
- CSV: adf_results, kpss_results, adf_lag_selection, rolling_summary

**Conclusión:**
- ADF/KPSS en close: Raíz unitaria no rechazada, estacionariedad rechazada.
- ADF/KPSS en log_close: Raíz unitaria no rechazada, estacionariedad rechazada.
- ADF/KPSS en r: Raíz unitaria rechazada, estacionariedad no rechazada (I(0)).
- ADF/KPSS en |r|: Raíz unitaria rechazada, estacionariedad rechazada por cambios de régimen.
- ADF/KPSS en log_rv_past_12: la serie no muestra una raiz unitaria clara, pero la estacionariedad es imperfecta por cambios de regimen; se conserva como serie principal por su persistencia y relevancia predictiva.

---

## FASE 4: Análisis de correlación y periodograma


**Actividades:**
- ACF r, |r|, log_rv_past_12 (lags 288, 2016)
- PACF r, |r|, log_rv_past_12
- Periodogramas (transformada Fourier)
- Búsqueda picos espectrales
- Potencia en referencias: 1h, 1día, 1 semana
- Lags principales: 1, 12, 288, 2016

**Resultados guardados:**
- SVG: 12 gráficos (ACF, PACF, periodogramas)
- CSV: acf_values, pacf_values, correlogram_summary, spectral_peaks, spectral_reference_power

**Conclusión:**
- ACF/PACF r: Autocorrelaciones insignificantes (lag 1 = -0.01), retornos impredecibles linealmente.
- ACF/PACF |r|: Significativas en lag 1 (0.34), evidencia de clustering de volatilidad.
- ACF/PACF log_rv_past_12: Persistencia fuerte (lag 1 = 0.98), estructura temporal bien definida.
- Periodogramas: Ciclos detectados en escala día/semana, sin indicios de comportamiento caótico.
- Conclusión global: log_rv_past_12 justificada para reconstrucción.

---

## FASE 5: Gráficos recurrencia de ventanas representativas

**Actividades:**
- Ventana 1 (quiet): baja volatilidad 
- Ventana 2 (high_volatility): máxima volatilidad
- Ventana 3 (recent): últimas observaciones
- Ventana 4 (middle): zona media
- Tamaño: 2,000 velas cada una
- Normalización z-score por ventana
- Distancia: absoluta 1D
- RR objetivo: 5%
- Series: r, |r|, log_rv_past_12

**Resultados guardados:**
- PNG: 12 recurrence plots
- CSV: selected_windows, recurrence_parameters

**Conclusión:**
- Recurrence plots r y |r|: Patrón fino cercano a ruido blanco.
- Recurrence plots log_rv_past_12: Patrón diagonal marcado indicando persistencia temporal con estructuras.
- Conclusión global: Patrones de persistencia en volatilidad; no prueba de no linealidad ni caos. La fase solo aporta evidencia visual exploratoria, especialmente en medidas de volatilidad.

---

## FASE 6: Filtrado lineal AR(p) (selección por BIC y check residuos)


**Actividades:**
- Split: train (2024-01-02 a 2025-06-30), test (2025-07-01 a 2026-04-30)
- Prueba p=1 a 100
- Selección por BIC: AR(49)
- Estimación Yule-Walker
- Cálculo residuos
- ACF/PACF residuos
- Test Ljung-Box residuos
- Recurrence plots residuos vs original

**Resultados guardados:**
- SVG: 5 gráficos (residuos temporales, histograma, ACF, PACF)
- PNG: 4 recurrence plots residuos
- CSV: ar_order_selection, ar_coefficients, residual_series, residual_statistics, ljung_box_residuals, etc.
- Modelo: AR(49) estandarizado

**Conclusión:**
- AR(49) seleccionado por BIC como modelo lineal óptimo.
- Residuos muestran autocorrelación residual mínima (|ACF(1)| = 0.003 vs 0.981 original).
- Ljung-Box en residuos rechaza solo a retardos altos, magnitud muy reducida.
- Recurrence plots residuos revelan textura dispersa pero estructura remanente persiste.
- Conclusión: La dependencia residual motiva los contrastes de no linealidad y la reconstrucción posterior, sin constituir por sí sola una prueba de caos.

---

## FASE 7: Contrastes no-linealidad (Ljung-Box, ARCH-LM, BDS)

**Actividades:**
- Ljung-Box sobre cuadrados z² y e² (lags 12, 24, 48, 96, 288, 2016)
- ARCH-LM sobre residuos (lags 12, 24, 48, 96, 288)
- BDS (Brock-Dechert-Scheinkman) en 4 ventanas
- BDS parámetros: m=[2,3,4], epsilon=[0.5, 1.0, 1.5]×sigma
- 50 permutaciones por ventana
- Comparación ACF cuadrados: original vs barajada

**Resultados guardados:**
- SVG: 4 gráficos (ACF cuadrados, BDS pvalues, shuffle ACF)
- CSV: ljungbox_squared, arch_lm, bds_results, shuffle_results, squared_acf_values, windows

**Conclusión:**
- Heterocedasticidad confirmada: Ljung-Box cuadrados rechaza en todos los retardos.
- ARCH-LM positivo en 5/5 retardos, dependencia en varianza clara.
- BDS rechaza en 36/36 combinaciones (serie original), 20/36 (residuos), indicando dependencia temporal.
- Comparación con barajadas: 7/12 estadísticos residuos en colas empíricas.
- Conclusión: Estructura dependiente del orden, embedding justificado.

---

## FASE 8: Reconstrucción espacio estados (τ=137, m=5)

**Actividades:**
- Información mutua media (AMI) para tau
- Primer mínimo local AMI: tau=137 velas
- Falsos vecinos cercanos (FNN): mínimo en m=5
- Método de Cao: inicio de estabilización en m=7; se mantiene m=5 como dimensión operativa
- Embedding: X_t = [z_t, z_(t-137), z_(t-274), z_(t-411), z_(t-548)]
- Cobertura temporal: 548 velas = 2,740 min ≈ 45.7 horas
- Train: 156,700 vectores
- Post-train: 87,504 vectores en `phase8_embedding_test.npz`
- Nota: 34,512 vectores corresponden al test final usado después en Fase 11, no al tramo post-train completo de Fase 8.
- Visualización 2D/3D
- Comparación AMI original vs barajada

**Resultados guardados:**
- SVG: 10 gráficos (AMI, FNN, Cao, embeddings 2D/3D)
- NPZ: phase8_embedding_train.npz, phase8_embedding_test.npz
- CSV: ami_tau, ami_shuffled_tau, fnn, cao, embedding_sample
- JSON: selected_embedding_params

**Conclusión:**
- AMI identifica tau=137 (11.4 horas) con diferencia clara respecto a barajada.
- FNN sugiere m=5 como mínimo práctico y Cao sitúa la meseta en m=7; ambos resultados son cercanos.
- Embedding genera 156,700 vectores train y 87,504 vectores post-train; el archivo `phase8_embedding_test.npz` no debe confundirse con el subconjunto final de test predictivo.
- Conclusión: Parámetros operativos establecidos sin garantía de atractor determinista.

---

## FASE 9: Cuantificación dinámica (D2, Lyapunov, entropía ordinal)

**Actividades:**
- Dimensión correlación D2 (Grassberger-Procaccia)
- Exponente Lyapunov aproximado (Rosenstein)
- Divergencia trayectorias cercanas
- Entropía permutación (órdenes 3-7)
- Retardos: 1 y tau=137
- Submuestreo: 2,500-3,000 vectores
- Comparación original vs serie barajada

**Resultados guardados:**
- SVG: 5 gráficos (D2 loglog, D2 slope, Lyapunov, PE, comparación)
- CSV: correlation_dimension, lyapunov_rosenstein, permutation_entropy
- JSON: correlation_dimension_summary, lyapunov_summary, quantification_summary

**Conclusión:**
- D2 = 3.83 (cercana a barajada 4.02), resultado ambiguo.
- Lyapunov = 0.0212/paso (vs barajada 0.00017), pendiente positiva sugiere divergencia.
- Entropía permutación = 0.895 (vs barajada 0.999), estructura ordinal presente.
- Conclusión: Indicios de estructura dinámica, no prueba concluyente de caos.

---

## FASE 10: Contrastes surrogados

**Frase:** Comparación vs barajadas, phase-randomized, AAFT

**Actividades específicas:**
- 50 series barajadas (conservan marginal, destrozan orden)
- 39 phase-randomized (conservan espectro)
- 39 AAFT (conservan marginal + estructura lineal)
- Cálculo D2, Lyapunov, entropía en cada variante
- Z-score original vs media surrogados
- Percentiles 5%, 50%, 95%
- P-values empíricos
- Boxplots por métrica

**Resultados guardados:**
- SVG: 7 gráficos (boxplots por métrica, histogramas)
- CSV: shuffled_stats, phase_randomized_stats, aaft_stats, surrogate_summary
- JSON: config, original_stats

**Conclusión:**
- Diferencia significativa vs barajadas en PE (S=469) y Lyapunov (S=54.1).
- Diferencia menor vs phase-randomized en Lyapunov/hora (S=0.054).
- Diferencia en PE delay=1 vs AAFT (S=2.51).
- Conclusión: Estructura temporal explotable, no necesariamente no lineal pura.

---

## FASE 11: Predicción local kNN (k=50)


**Actividades:**
- Split: train (2024-01 a 2025-06), validation (2025-07 a 2025-12), test (2026-01 a 2026-04)
- Modelos:
  * Media histórica
  * Persistencia
  * AR(49) forecast recursivo
  * 1-NN
  * k-NN: k=[2, 3, 5, 10, 20, 50]
- Búsqueda vecinos solo en histórico
- Exclusión Theiler window (685 retardos)
- Validación no-leakage
- Selección k=50 por RMSE validación
- Test: 5,000 muestras
- Métricas: MAE, MSE, RMSE, R² OOS, sesgo

**Resultados guardados:**
- SVG: 5 gráficos (k selection, real vs pred, errores temporal, histograma, comparación)
- CSV: validation_k_selection, test_metrics, predictions_test_sample, split_summary
- JSON: prediction_summary

**Conclusión:**
- kNN con k=50 logra RMSE test = 0.893877 vs persistencia 0.9775 (~8.6% mejora).
- AR(49) obtiene mejor desempeño: RMSE = 0.864067.
- Conclusión: Predictibilidad local confirmada sin requerir dinámicas caóticas.

---

## FASE 12: Validación robustez de parámetros (k, m, τ)

**Actividades:**
- Extensión Fase 11 sin modelos nuevos
- Verificar si k=50 cierra rejilla superior
- Usar m=14 como contraste deliberado de alta dimensión frente a m=5
- Configuración tau137_m5 → memoria 45.7h, 5000 puntos evaluación
- Configuración tau137_m14 → memoria 148.4h (6.2 días), 1000 puntos (coste computacional)
- Búsqueda k extendida: [2, 3, 5, 10, 20, 50, 100, 200]
- Reglas anti-leakage mantenidas estrictamente
- Test evaluado una sola vez por configuración con k seleccionado

**Conclusión:**
- Mejor k para m=5: k=200 (mejora marginal vs k=100, estabilización).
- Mejor k para m=14: k=200 (peor rendimiento global que m=5).
- m=14 no mejora a m=5 en test (curse of dimensionality, menor densidad local).
- AR(49) sigue siendo mejor modelo que kNN en ambas configuraciones.
- Conclusión: m=5 confirmada como práctica, k=50-200 ofrecen rendimiento similar.

**Resultados guardados:**
- SVG: 4 gráficos (validation RMSE m5, validation RMSE m14, comparación test y ventana real vs predicho)
- CSV: validation_k_selection (m5 y m14), test_metrics (m5 y m14), config_comparison
- JSON: prediction_summary

---

## FASE 13: Validación metodológica en mapa logístico caótico

**Actividades:**
- Sistema sintético: mapa logístico x[t+1] = 4.0·x[t]·(1-x[t]), determinista caótico
- x0=0.123456789, n_total=12000, burn_in=1000, 11000 observaciones útiles
- 3 variantes: limpia, ruido pequeño (σ=0.00355), ruido moderado (σ=0.01777)
- AMI: τ=9 (limpia/pequeño), τ=5 (moderado) - ruido cambia escala temporal
- FNN/Cao: m=5-6 (FNN) vs m=10 (Cao) - discrepancia metodológica clara
- Embedding: 2D/3D visualización, estructura más clara que BTC
- Lyapunov Rosenstein: pendiente positiva en la serie limpia y en la serie con ruido pequeño; la serie con ruido moderado pierde una región de crecimiento interpretable.
- Entropía/permutación no se usa aquí como contraste con barajadas; Fase 13 es un control sintético limpio/ruidoso, separado de las Fases 9-10.
- Predicción: AR lineal seleccionado por BIC con candidatos p=0..100; en Fase 13 selecciona AR(0) / media histórica
- kNN medio mejor que persistencia y AR(0) en las tres series
- MAE test kNN: limpia 0.0745, pequeño 0.1006, moderado 0.0823
- RMSE test: limpia AR(0)=0.3565 vs kNN=0.0982; pequeño AR(0)=0.3564 vs kNN=0.1301; moderado AR(0)=0.3558 vs kNN=0.1075

**Conclusión:**
- Pipeline funciona bien en caos sintético limpio (RMSE kNN=0.098 frente a AR(0)=0.357).
- Ruido degrada reconstrucción y predicción, pero no destruye estructura.
- En el mapa logístico, la ventaja del kNN frente al AR lineal no se debe a usar más retardos, sino a capturar una transición local no lineal.
- En BTC el AR(49) sigue siendo muy competitivo; los datos reales mezclan dependencia lineal, ruido, heterocedasticidad y cambios de régimen.
- Conclusión: Metodología validada en sistema controlado, interpretación prudente en BTC.

**Resultados guardados:**
- SVG: series sintéticas, AMI, FNN, Cao, embeddings 2D/3D, mapa de transición, BIC AR, predicciones compactas/completas, métricas
- CSV: ami.csv, fnn.csv, cao.csv, embedding_params, lyapunov_rosenstein, ar_order_selection, prediction_metrics, validation_k_selection
- CSV: `phase13_selected_embedding_params.csv`, `phase13_lyapunov_summary.csv` y demás tablas de fase.
- JSON: `phase13_prediction_summary.json`
- Datos: `data/synthetic/logistic_clean.csv`, `data/synthetic/logistic_noise_small.csv`, `data/synthetic/logistic_noise_moderate.csv`

---

## FASE 14: HAR-logRV compacto y exportable para MVP

**Actividades:**
- Modelo nuevo único: HAR-logRV compacto por OLS.
- Target mantenido: `log_rv_future_12`.
- Features usadas: `log_rv_past_12`, `log_rv_past_48`, `log_rv_past_288`.
- Split temporal igual que Fases 11/12: train hasta 2025-06-30, validation hasta 2025-12-31, test desde 2026-01-01.
- Comparación contra media histórica, persistencia, AR(49) horizonte 12 y kNN `tau=137`, `m=5`, `k=200`.
- Evaluación comparable en 5000 puntos de test para incluir kNN.
- Evaluación adicional `test_full` para HAR, AR(49), persistencia y media histórica.
- Simulación MVP con últimas 1000 velas, split interno 70/30.
- Exportación de artefacto `data/model_artifacts/har_logrv_model.json`.

**Conclusión:**
- HAR-logRV mejora a persistencia.
- En la muestra comparable, HAR-logRV queda ligeramente por delante de AR(49): RMSE 0.8608 vs 0.8641.
- HAR-logRV queda por delante de kNN `tau137_m5_k200`: RMSE 0.8608 vs 0.8765.
- El resultado refuerza que la predictibilidad de BTC se explica bien mediante memoria multiescala de volatilidad realizada.
- HAR-logRV es razonable como modelo práctico principal del MVP por simplicidad, velocidad e interpretabilidad.
- La reconstrucción no lineal queda como análisis metodológico; HAR-logRV queda como cierre predictivo operativo.

**Resultados guardados:**
- SVG: comparación métricas general, comparación dedicada AR(49)/kNN/HAR, real vs predicho, distribución de error absoluto AR(49)/kNN/HAR, errores temporales, histograma de errores, coeficientes HAR, contexto MVP 1000 velas.
- CSV: har_coefficients, validation_metrics, test_metrics, predictions_test_comparable, predictions_test_sample, model_comparison, mvp_1000_mode_summary.
- JSON: phase14_har_model_artifact, phase14_prediction_summary.
- Artefacto MVP: `data/model_artifacts/har_logrv_model.json`.

---

**Modelos finales:** AR(49) referencia lineal, Embedding(τ=137, m=5), kNN(k=50-200) predictor local, HAR-logRV compacto como modelo práctico exportable.
