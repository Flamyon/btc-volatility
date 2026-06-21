# DATA_MANIFEST

Manifiesto de datos utilizado en el repositorio técnico `btc-volatility`.

Este documento identifica la fuente de datos, el periodo temporal utilizado, los ficheros brutos descargados y los datasets derivados.

## Fuente oficial

Los datos proceden de Binance Spot, par `BTCUSDT`, intervalo `5m`, mediante los ficheros públicos mensuales de klines.

Ruta oficial:

```text
https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/
```

Patrón de ficheros:

```text
BTCUSDT-5m-YYYY-MM.zip
```

Cada fichero ZIP contiene un CSV mensual de velas OHLCV de Binance para el par `BTCUSDT` y frecuencia de cinco minutos.

## Periodo utilizado

Periodo temporal cubierto por el dataset limpio:

```text
2024-01-01 00:00:00 UTC -- 2026-04-30 23:55:00 UTC
```

Meses descargados:

```text
2024-01
2024-02
2024-03
2024-04
2024-05
2024-06
2024-07
2024-08
2024-09
2024-10
2024-11
2024-12
2025-01
2025-02
2025-03
2025-04
2025-05
2025-06
2025-07
2025-08
2025-09
2025-10
2025-11
2025-12
2026-01
2026-02
2026-03
2026-04
```

## Fecha de descarga

Fecha de descarga de los ficheros brutos:

```text
<2026-05-08>
```

## Ficheros brutos

Ubicación esperada dentro del repositorio:

```text
data/raw/
```

Ficheros ZIP utilizados:

```text
data/raw/BTCUSDT-5m-2024-01.zip
data/raw/BTCUSDT-5m-2024-02.zip
data/raw/BTCUSDT-5m-2024-03.zip
data/raw/BTCUSDT-5m-2024-04.zip
data/raw/BTCUSDT-5m-2024-05.zip
data/raw/BTCUSDT-5m-2024-06.zip
data/raw/BTCUSDT-5m-2024-07.zip
data/raw/BTCUSDT-5m-2024-08.zip
data/raw/BTCUSDT-5m-2024-09.zip
data/raw/BTCUSDT-5m-2024-10.zip
data/raw/BTCUSDT-5m-2024-11.zip
data/raw/BTCUSDT-5m-2024-12.zip
data/raw/BTCUSDT-5m-2025-01.zip
data/raw/BTCUSDT-5m-2025-02.zip
data/raw/BTCUSDT-5m-2025-03.zip
data/raw/BTCUSDT-5m-2025-04.zip
data/raw/BTCUSDT-5m-2025-05.zip
data/raw/BTCUSDT-5m-2025-06.zip
data/raw/BTCUSDT-5m-2025-07.zip
data/raw/BTCUSDT-5m-2025-08.zip
data/raw/BTCUSDT-5m-2025-09.zip
data/raw/BTCUSDT-5m-2025-10.zip
data/raw/BTCUSDT-5m-2025-11.zip
data/raw/BTCUSDT-5m-2025-12.zip
data/raw/BTCUSDT-5m-2026-01.zip
data/raw/BTCUSDT-5m-2026-02.zip
data/raw/BTCUSDT-5m-2026-03.zip
data/raw/BTCUSDT-5m-2026-04.zip
```

## Esquema original de Binance

Los ficheros mensuales de klines de Binance contienen columnas OHLCV sin cabecera. El esquema usado por el script de limpieza es:

```text
open_time
open
high
low
close
volume
close_time
quote_asset_volume
number_of_trades
taker_buy_base_asset_volume
taker_buy_quote_asset_volume
ignore
```

En el repositorio, estas columnas se transforman y filtran durante la construcción del dataset limpio.

## Dataset limpio

Fichero limpio principal:

```text
btc_5m_clean.csv
```

Rango temporal validado:

```text
2024-01-01 00:00:00 UTC -- 2026-04-30 23:55:00 UTC
```

Número de observaciones:

```text
245088
```

Frecuencia:

```text
5 minutos
```

Validaciones realizadas:

```text
columnas esperadas: OK
rango temporal: OK
frecuencia modal de 5 minutos: OK
open_time estrictamente creciente: OK
duplicados: 0
huecos temporales: 0
valores nulos: 0
precios positivos: OK
```

## Scripts relacionados

Construcción del dataset limpio:

```text
src/build_btc_5m_clean.py
```

Validación integral del dataset limpio:

```text
src/validate.py
```

Construcción de variables de volatilidad realizada:

```text
src/build_phase1_features.py
```

## Advertencia

Los datos originales proceden de una fuente pública externa. Este repositorio documenta el procedimiento de descarga, limpieza, validación y transformación utilizado en el TFG, pero no garantiza que la fuente externa conserve indefinidamente los mismos ficheros o checksums.