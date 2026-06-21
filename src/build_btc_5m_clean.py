"""
Construye un dataset limpio de velas BTCUSDT 5m de Binance Spot.

Entrada esperada:
    ZIPs mensuales descargados de Binance en ./data/raw, por ejemplo:
    BTCUSDT-5m-2024-01.zip, BTCUSDT-5m-2024-02.zip, ...
    https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2024-01.zip

Salida:
    btc_5m_clean.csv con columnas:
    open_time, open, high, low, close, volume, trades, taker_buy_quote

2024: 366 días
2025: 365 días
2026 enero-abril: 120 días
Total: 851 días
851 × 288 velas de 5 minutos = 245.088 velas    
"""

import glob
import io
import os
import zipfile


try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: falta la dependencia 'pandas'.\n"
    ) from exc



EXPECTED_FREQUENCY = pd.Timedelta(minutes=5)


# Columnas originales de los CSV de Binance. Los CSV vienen sin cabecera.
RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


# Columnas finales requeridas, en el orden exacto del dataset limpio.
FINAL_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
    "taker_buy_quote",
]


NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_buy_quote",
]


def convert_open_time_to_datetime(open_time: pd.Series) -> pd.Series:
    """
    Convierte open_time Unix a datetime UTC sin zona horaria.

    Binance suele publicar klines en milisegundos, pero algunos ZIPs historicos
    pueden venir en microsegundos. Detectamos la unidad por magnitud para evitar
    fechas imposibles como el ano 58299 al interpretar microsegundos como ms.
    """
    values = pd.to_numeric(open_time, errors="coerce")
    values_ms = values.astype("float64")

    microseconds_mask = values.abs().between(10**14, 10**17, inclusive="left")
    nanoseconds_mask = values.abs() >= 10**17

    values_ms.loc[microseconds_mask] = values.loc[microseconds_mask] / 1_000
    values_ms.loc[nanoseconds_mask] = values.loc[nanoseconds_mask] / 1_000_000

    return pd.to_datetime(
        values_ms,
        unit="ms",
        utc=True,
        errors="coerce",
    ).dt.tz_localize(None)


def warn(message: str) -> None:
    """Imprime advertencias sin detener el procesamiento completo."""
    print(f"ADVERTENCIA: {message}")


def read_single_zip(zip_path: str) -> pd.DataFrame | None:
    """
    Lee el CSV interno de un ZIP de Binance y devuelve un DataFrame.

    Si el ZIP esta corrupto, vacio o no contiene CSV, devuelve None para que el
    flujo principal pueda continuar con el resto de archivos.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            csv_files = sorted(
                name
                for name in zip_file.namelist()
                if name.lower().endswith(".csv") and not name.endswith("/")
            )

            if not csv_files:
                warn(f"{zip_path} no contiene ningun CSV. Se omite.")
                return None

            if len(csv_files) > 1:
                warn(
                    f"{zip_path} contiene mas de un CSV. "
                    f"Se usara el primero: {csv_files[0]}"
                )

            csv_name = csv_files[0]
            csv_bytes = zip_file.read(csv_name)

        return pd.read_csv(
            io.BytesIO(csv_bytes),
            header=None,
            names=RAW_COLUMNS,
        )

    except Exception as exc:
        warn(f"Error inesperado leyendo {zip_path}: {exc}. Se omite.")

    return None


def load_all_zips(input_dir: str) -> pd.DataFrame:
    """Lee todos los ZIPs de input_dir en orden alfabetico y los concatena."""
    zip_paths = sorted(glob.glob(os.path.join(input_dir, "*.zip")))

    if not zip_paths:
        raise FileNotFoundError(f"No se encontraron archivos .zip en {input_dir}")

    frames: list[pd.DataFrame] = []

    print(f"ZIPs encontrados: {len(zip_paths)}")
    for zip_path in zip_paths:
        print(f"Leyendo: {zip_path}")
        frame = read_single_zip(zip_path)
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No se pudo leer ningun CSV valido desde los ZIPs.")

    return pd.concat(frames, ignore_index=True)


def clean_klines(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los datos siguiendo las reglas solicitadas.

    - Convierte open_time de Unix ms a datetime UTC sin zona horaria.
    - Ordena por open_time.
    - Elimina duplicados por open_time, conservando la primera ocurrencia.
    - Convierte columnas numericas a tipos numericos.
    - Selecciona solo las columnas finales.
    """
    df = raw_df.copy()

    # Binance publica los tiempos en UTC. Guardamos datetime naive en UTC para
    # que el CSV sea sencillo de leer en pandas, Excel u otras herramientas.
    df["open_time"] = convert_open_time_to_datetime(df["open_time"])

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # trades deberia ser entero; usamos Int64 nullable para no romper si aparece
    # algun valor invalido que se convierta en NA.
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")

    df = df.sort_values("open_time", kind="mergesort")
    df = df.drop_duplicates(subset="open_time", keep="first")
    df = df[FINAL_COLUMNS].reset_index(drop=True)

    return df


def count_duplicate_open_times(raw_df: pd.DataFrame) -> int:
    """Cuenta timestamps duplicados antes de eliminar duplicados."""
    open_times = convert_open_time_to_datetime(raw_df["open_time"])
    return int(open_times.duplicated().sum())


def build_gap_table(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Calcula gaps mayores a la frecuencia esperada.

    Devuelve:
    - DataFrame con ejemplos de gaps.
    - Serie con todas las diferencias entre timestamps consecutivos.
    """
    valid_times = clean_df["open_time"].dropna().sort_values().reset_index(drop=True)
    time_diffs = valid_times.diff()
    gap_mask = time_diffs > EXPECTED_FREQUENCY

    if not gap_mask.any():
        return pd.DataFrame(), time_diffs.dropna()

    gap_positions = gap_mask[gap_mask].index
    gaps = pd.DataFrame(
        {
            "previous_open_time": valid_times.iloc[gap_positions - 1].to_numpy(),
            "next_open_time": valid_times.iloc[gap_positions].to_numpy(),
            "gap_duration": time_diffs.iloc[gap_positions].to_numpy(),
        }
    )

    gaps["missing_5m_candles"] = (
        gaps["gap_duration"] / EXPECTED_FREQUENCY - 1
    ).astype("int64")

    return gaps, time_diffs.dropna()


def print_quality_summary(
    original_rows: int,
    duplicate_timestamps_before: int,
    clean_df: pd.DataFrame,
) -> None:
    """Imprime el control de calidad antes de guardar el CSV limpio."""
    gaps, time_diffs = build_gap_table(clean_df)
    null_percentages = clean_df.isna().mean().mul(100)
    duplicated_after_cleaning = int(clean_df["open_time"].duplicated().sum())

    if time_diffs.empty:
        mean_diff = pd.NaT
        median_diff = pd.NaT
        all_diffs_are_expected = True
    else:
        mean_diff = time_diffs.mean()
        median_diff = time_diffs.median()
        all_diffs_are_expected = bool((time_diffs == EXPECTED_FREQUENCY).all())

    series_is_regular = (
        duplicated_after_cleaning == 0
        and clean_df["open_time"].notna().all()
        and gaps.empty
        and all_diffs_are_expected
    )

    print("\n" + "=" * 72)
    print("CONTROL DE CALIDAD - BTCUSDT 5m")
    print("=" * 72)
    print(f"Velas originales antes de limpieza: {original_rows:,}")
    print(f"Velas despues de limpieza:          {len(clean_df):,}")
    print(f"Rango temporal minimo:              {clean_df['open_time'].min()}")
    print(f"Rango temporal maximo:              {clean_df['open_time'].max()}")

    print("\nPorcentaje de valores nulos por columna:")
    for column, percentage in null_percentages.items():
        print(f"  - {column}: {percentage:.6f}%")

    print(f"\nTimestamps duplicados antes de limpiar: {duplicate_timestamps_before:,}")
    print(f"Timestamps duplicados tras limpieza:    {duplicated_after_cleaning:,}")
    print(f"Frecuencia esperada:                 {EXPECTED_FREQUENCY}")
    print(f"Diferencia media entre timestamps:   {mean_diff}")
    print(f"Diferencia mediana entre timestamps: {median_diff}")
    print(f"Numero de gaps > 5 minutos:          {len(gaps):,}")
    print(f"Serie regular a 5 minutos:           {'Si' if series_is_regular else 'No'}")

    if gaps.empty:
        print("\nNo se detectaron gaps mayores a 5 minutos.")
    else:
        print(f"\nPrimeros {min(10, len(gaps))} gaps detectados:")
        print(gaps.head(10).to_string(index=False))

    print("=" * 72 + "\n")


def ensure_output_directory(output_csv: str) -> None:
    """Crea el directorio de salida si OUTPUT_CSV incluye una carpeta."""
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def main() -> None:
    """Punto de entrada del script."""
    raw_df = load_all_zips("./data/raw")
    original_rows = len(raw_df)
    duplicate_timestamps_before = count_duplicate_open_times(raw_df)

    clean_df = clean_klines(raw_df)
    print_quality_summary(original_rows, duplicate_timestamps_before, clean_df)

    ensure_output_directory("btc_5m_clean.csv")
    clean_df.to_csv("btc_5m_clean.csv", index=False)
    print("CSV limpio guardado en: btc_5m_clean.csv")


if __name__ == "__main__":
    main()
