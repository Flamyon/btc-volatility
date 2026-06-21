# Fase 1 - Construccion de variables

Dataset procesado: `data/processed/btc_5m_features.csv`

## Shape final

| metric | value |
| --- | --- |
| input_rows | 245088 |
| output_rows_after_dropna | 244752 |
| rows_dropped_total | 336 |
| rows_dropped_start | 288 |
| rows_dropped_end | 48 |
| first_valid_index_zero_based | 288 |
| last_valid_index_zero_based | 245039 |
| final_start_open_time | 2024-01-02 00:00:00 |
| final_end_open_time | 2026-04-30 19:55:00 |
| epsilon_for_log_rv | 1e-12 |

## Columnas creadas

| column | definition |
| --- | --- |
| log_close | log(close) |
| r | log(close_t) - log(close_{t-1}) |
| r2 | r_t^2 |
| abs_r | \|r_t\| |
| hl_range | log(high / low) |
| log_volume | log(volume + 1) |
| log_trades | log(trades + 1) |
| rv_past_12 | sum_{i=0}^{11} r_{t-i}^2 |
| rv_past_48 | sum_{i=0}^{47} r_{t-i}^2 |
| rv_past_288 | sum_{i=0}^{287} r_{t-i}^2 |
| rv_future_12 | sum_{i=1}^{12} r_{t+i}^2 |
| rv_future_48 | sum_{i=1}^{48} r_{t+i}^2 |
| log_rv_past_12 | log(rv_past_12 + epsilon) |
| log_rv_past_48 | log(rv_past_48 + epsilon) |
| log_rv_past_288 | log(rv_past_288 + epsilon) |
| log_rv_future_12 | log(rv_future_12 + epsilon) |
| log_rv_future_48 | log(rv_future_48 + epsilon) |

## Primeras filas tras dropna

| open_time | close | r | abs_r | rv_past_12 | rv_future_12 | log_rv_past_12 | log_rv_future_12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01-02 00:00:00 | 44264.6 | 0.00192393 | 0.00192393 | 3.21021e-05 | 0.000119108 | -10.3466 | -9.03548 |
| 2024-01-02 00:05:00 | 44398.4 | 0.0030184 | 0.0030184 | 4.08271e-05 | 0.000111846 | -10.1062 | -9.09839 |
| 2024-01-02 00:10:00 | 44630.1 | 0.00520396 | 0.00520396 | 6.77101e-05 | 0.000144591 | -9.60027 | -8.8416 |
| 2024-01-02 00:15:00 | 44555.9 | -0.00166394 | 0.00166394 | 6.8443e-05 | 0.000148951 | -9.58951 | -8.8119 |
| 2024-01-02 00:20:00 | 44751.9 | 0.00439022 | 0.00439022 | 8.39587e-05 | 0.00012972 | -9.38519 | -8.95013 |

## Ultimas filas tras dropna

| open_time | close | r | abs_r | rv_past_12 | rv_future_12 | log_rv_past_12 | log_rv_future_12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-30 19:35:00 | 76493.1 | 0.000134269 | 0.000134269 | 4.33134e-06 | 2.33992e-06 | -12.3496 | -12.9654 |
| 2026-04-30 19:40:00 | 76443.3 | -0.000651905 | 0.000651905 | 3.32303e-06 | 1.9153e-06 | -12.6146 | -13.1656 |
| 2026-04-30 19:45:00 | 76442.7 | -7.32572e-06 | 7.32572e-06 | 3.2635e-06 | 2.22777e-06 | -12.6327 | -13.0145 |
| 2026-04-30 19:50:00 | 76415 | -0.000362559 | 0.000362559 | 3.31731e-06 | 2.21815e-06 | -12.6164 | -13.0188 |
| 2026-04-30 19:55:00 | 76412.8 | -2.87906e-05 | 2.87906e-05 | 2.95537e-06 | 2.56045e-06 | -12.7319 | -12.8753 |

## Perdida de observaciones

Se eliminan 288 filas iniciales porque `rv_past_288` necesita 288 retornos pasados validos. El primer retorno de la muestra no existe porque requiere el cierre anterior.

Se eliminan 48 filas finales porque `rv_future_48` necesita 48 retornos estrictamente posteriores. El target futuro nunca incluye `r_t`; empieza en `t+1`.

## Control de leakage

`rv_past_*` se calcula con ventanas inclusivas hasta `t`, mientras que `rv_future_*` se calcula con ventanas desde `t+1` hasta el horizonte correspondiente. Por tanto, las variables explicativas basadas en pasado no usan informacion futura.

## Conclusion parcial

La Fase 1 deja un dataframe utilizable para caracterizacion y prediccion de volatilidad. La serie principal recomendada para las fases siguientes es `log_rv_past_12`, y el target principal es `log_rv_future_12`.
