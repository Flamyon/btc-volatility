# Fase 0 - Validacion y limpieza de datos

Archivo validado: `btc_5m_clean.csv`

## Origen de los datos

El dataset parte de los ZIP mensuales oficiales de Binance Spot para el par `BTCUSDT` con frecuencia de 5 minutos. Los ficheros brutos se conservan en `data/raw/` con nombres del tipo `BTCUSDT-5m-2024-01.zip` y proceden de la ruta pública `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/`. El script `src/build_btc_5m_clean.py` lee los CSV internos de cada ZIP, normaliza las columnas originales de Binance y genera `btc_5m_clean.csv`.

## Tabla resumen de calidad

| check | value | status | detail |
| --- | --- | --- | --- |
| Columnas esperadas | True | OK | open_time, open, high, low, close, volume, trades, taker_buy_quote |
| Numero de filas | 245088 | OK | esperadas=245088 |
| Fecha minima | 2024-01-01 00:00:00 | OK | esperada=2024-01-01 00:00:00 |
| Fecha maxima | 2026-04-30 23:55:00 | OK | esperada=2026-04-30 23:55:00 |
| Orden temporal en el CSV | True | OK | open_time estrictamente creciente |
| Frecuencia real modal | 0:05:00 | OK | observaciones=245087 |
| Diferencias distintas de 5m | 0 | OK |  |
| Duplicados en open_time | 0 | OK | [] |
| Huecos temporales > 5m | 0 | OK | velas faltantes=0 |
| Valores nulos/blancos | 0 | OK | {} |
| Errores de conversion numerica/fecha | 0 | OK | {} |
| Precios <= 0 | 0 | OK | open/high/low/close |
| Volumen <= 0 | 0 | OK | ceros=0, negativos=0 |
| Trades <= 0 | 0 | OK | ceros=0, negativos=0 |
| Taker buy quote <= 0 | 0 | OK | ceros=0, negativos=0 |
| Consistencia OHLC | 0 | OK | {} |
| Velas esperadas segun rango observado | 245088 | OK | rango observado inclusivo |

## Resumen temporal

- Numero de filas: 245,088
- Fecha minima: 2024-01-01 00:00:00
- Fecha maxima: 2026-04-30 23:55:00
- Velas esperadas en el rango declarado: 245,088
- Velas esperadas segun el rango observado: 245,088
- Duplicados en `open_time`: 0
- Huecos temporales: 0
- Velas de 5 minutos faltantes: 0

## Interpretacion

La serie cubre exactamente el rango declarado, con una vela cada cinco minutos, sin duplicados, sin huecos temporales, sin nulos y sin valores no positivos en precios, volumen, trades o taker_buy_quote. Por tanto, la base de datos es utilizable para construir retornos y volatilidad realizada sin necesidad de imputacion temporal previa.

## Conclusion parcial

El dataset es valido para continuar con la Fase 1.
