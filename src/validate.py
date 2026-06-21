"""
Fase 0: validacion del dataset BTCUSDT 5m.

Este script valida el CSV limpio antes de construir variables de volatilidad.
Usa solo libreria estandar para que la comprobacion pueda ejecutarse incluso
en entornos donde todavia no este instalado pandas.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


EXPECTED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
    "taker_buy_quote",
]

PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "trades", "taker_buy_quote"]
EXPECTED_FREQUENCY = timedelta(minutes=5)
DEFAULT_EXPECTED_START = datetime(2024, 1, 1, 0, 0, 0)
DEFAULT_EXPECTED_END = datetime(2026, 4, 30, 23, 55, 0)


@dataclass(frozen=True)
class Gap:
    previous_open_time: datetime
    next_open_time: datetime
    gap_duration: timedelta
    missing_5m_candles: int


def parse_datetime(value: str) -> datetime:
    """Parsea timestamps ISO como los guardados en btc_5m_clean.csv."""
    return datetime.fromisoformat(value)


def parse_decimal(value: str) -> Decimal:
    """Convierte texto numerico a Decimal para evitar errores de coma flotante."""
    return Decimal(str(value))


def expected_candle_count(start: datetime, end: datetime) -> int:
    """Numero de velas esperadas en un rango inclusivo con frecuencia 5m."""
    return int((end - start) / EXPECTED_FREQUENCY) + 1


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Escribe una tabla CSV, manteniendo cabecera aunque no haya filas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_console_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Construye una tabla compacta para revisar la validacion en consola."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def validate_csv(
    csv_path: Path,
    expected_start: datetime = DEFAULT_EXPECTED_START,
    expected_end: datetime = DEFAULT_EXPECTED_END,
) -> dict[str, Any]:
    """Valida estructura, frecuencia temporal y calidad numerica del CSV."""
    rows = 0
    header: list[str] | None = None
    timestamps: list[datetime] = []
    seen_timestamps: set[datetime] = set()
    duplicate_timestamps = 0
    duplicate_examples: list[tuple[int, str]] = []
    null_counts: Counter[str] = Counter()
    parse_errors: Counter[str] = Counter()
    non_positive: Counter[str] = Counter()
    zero_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    logical_errors: Counter[str] = Counter()
    input_is_monotonic = True
    previous_input_time: datetime | None = None

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames

        for line_number, row in enumerate(reader, start=2):
            rows += 1

            for column in EXPECTED_COLUMNS:
                value = row.get(column)
                if value is None or str(value).strip() == "":
                    null_counts[column] += 1

            timestamp: datetime | None = None
            try:
                timestamp = parse_datetime(row["open_time"])
                timestamps.append(timestamp)
            except Exception:
                parse_errors["open_time"] += 1

            if timestamp is not None:
                if previous_input_time is not None and timestamp <= previous_input_time:
                    input_is_monotonic = False
                previous_input_time = timestamp

                if timestamp in seen_timestamps:
                    duplicate_timestamps += 1
                    if len(duplicate_examples) < 10:
                        duplicate_examples.append((line_number, str(timestamp)))
                seen_timestamps.add(timestamp)

            parsed_values: dict[str, Decimal] = {}
            for column in NUMERIC_COLUMNS:
                try:
                    parsed = parse_decimal(row[column])
                    parsed_values[column] = parsed
                except (InvalidOperation, KeyError):
                    parse_errors[column] += 1
                    continue

                if parsed < 0:
                    negative_counts[column] += 1
                if parsed == 0:
                    zero_counts[column] += 1
                if parsed <= 0:
                    non_positive[column] += 1

            try:
                open_price = parsed_values["open"]
                high_price = parsed_values["high"]
                low_price = parsed_values["low"]
                close_price = parsed_values["close"]
            except KeyError:
                continue

            if high_price < low_price:
                logical_errors["high_lt_low"] += 1
            if not low_price <= open_price <= high_price:
                logical_errors["open_outside_high_low"] += 1
            if not low_price <= close_price <= high_price:
                logical_errors["close_outside_high_low"] += 1

    sorted_timestamps = sorted(timestamps)
    min_time = sorted_timestamps[0] if sorted_timestamps else None
    max_time = sorted_timestamps[-1] if sorted_timestamps else None

    diffs = Counter(
        sorted_timestamps[index] - sorted_timestamps[index - 1]
        for index in range(1, len(sorted_timestamps))
    )

    gaps: list[Gap] = []
    for index in range(1, len(sorted_timestamps)):
        previous_time = sorted_timestamps[index - 1]
        current_time = sorted_timestamps[index]
        diff = current_time - previous_time
        if diff > EXPECTED_FREQUENCY:
            gaps.append(
                Gap(
                    previous_open_time=previous_time,
                    next_open_time=current_time,
                    gap_duration=diff,
                    missing_5m_candles=int(diff / EXPECTED_FREQUENCY) - 1,
                )
            )

    expected_declared = expected_candle_count(expected_start, expected_end)
    expected_observed = (
        expected_candle_count(min_time, max_time)
        if min_time is not None and max_time is not None
        else None
    )
    non_expected_diffs = sum(count for diff, count in diffs.items() if diff != EXPECTED_FREQUENCY)
    missing_5m_candles = sum(gap.missing_5m_candles for gap in gaps)
    total_nulls = sum(null_counts.values())
    total_parse_errors = sum(parse_errors.values())
    non_positive_prices = sum(non_positive[column] for column in PRICE_COLUMNS)
    total_logical_errors = sum(logical_errors.values())
    frequency_modal = diffs.most_common(1)[0] if diffs else (None, 0)

    quality_rows = [
        {
            "check": "Columnas esperadas",
            "value": header == EXPECTED_COLUMNS,
            "status": "OK" if header == EXPECTED_COLUMNS else "REVISAR",
            "detail": ", ".join(header or []),
        },
        {
            "check": "Numero de filas",
            "value": rows,
            "status": "OK" if rows == expected_declared else "REVISAR",
            "detail": f"esperadas={expected_declared}",
        },
        {
            "check": "Fecha minima",
            "value": min_time,
            "status": "OK" if min_time == expected_start else "REVISAR",
            "detail": f"esperada={expected_start}",
        },
        {
            "check": "Fecha maxima",
            "value": max_time,
            "status": "OK" if max_time == expected_end else "REVISAR",
            "detail": f"esperada={expected_end}",
        },
        {
            "check": "Orden temporal en el CSV",
            "value": input_is_monotonic,
            "status": "OK" if input_is_monotonic else "REVISAR",
            "detail": "open_time estrictamente creciente",
        },
        {
            "check": "Frecuencia real modal",
            "value": str(frequency_modal[0]),
            "status": "OK" if frequency_modal[0] == EXPECTED_FREQUENCY else "REVISAR",
            "detail": f"observaciones={frequency_modal[1]}",
        },
        {
            "check": "Diferencias distintas de 5m",
            "value": non_expected_diffs,
            "status": "OK" if non_expected_diffs == 0 else "REVISAR",
            "detail": "",
        },
        {
            "check": "Duplicados en open_time",
            "value": duplicate_timestamps,
            "status": "OK" if duplicate_timestamps == 0 else "REVISAR",
            "detail": duplicate_examples,
        },
        {
            "check": "Huecos temporales > 5m",
            "value": len(gaps),
            "status": "OK" if len(gaps) == 0 else "REVISAR",
            "detail": f"velas faltantes={missing_5m_candles}",
        },
        {
            "check": "Valores nulos/blancos",
            "value": total_nulls,
            "status": "OK" if total_nulls == 0 else "REVISAR",
            "detail": dict(null_counts),
        },
        {
            "check": "Errores de conversion numerica/fecha",
            "value": total_parse_errors,
            "status": "OK" if total_parse_errors == 0 else "REVISAR",
            "detail": dict(parse_errors),
        },
        {
            "check": "Precios <= 0",
            "value": non_positive_prices,
            "status": "OK" if non_positive_prices == 0 else "REVISAR",
            "detail": "open/high/low/close",
        },
        {
            "check": "Volumen <= 0",
            "value": non_positive["volume"],
            "status": "OK" if non_positive["volume"] == 0 else "REVISAR",
            "detail": f"ceros={zero_counts['volume']}, negativos={negative_counts['volume']}",
        },
        {
            "check": "Trades <= 0",
            "value": non_positive["trades"],
            "status": "OK" if non_positive["trades"] == 0 else "REVISAR",
            "detail": f"ceros={zero_counts['trades']}, negativos={negative_counts['trades']}",
        },
        {
            "check": "Taker buy quote <= 0",
            "value": non_positive["taker_buy_quote"],
            "status": "OK" if non_positive["taker_buy_quote"] == 0 else "REVISAR",
            "detail": (
                f"ceros={zero_counts['taker_buy_quote']}, "
                f"negativos={negative_counts['taker_buy_quote']}"
            ),
        },
        {
            "check": "Consistencia OHLC",
            "value": total_logical_errors,
            "status": "OK" if total_logical_errors == 0 else "REVISAR",
            "detail": dict(logical_errors),
        },
        {
            "check": "Velas esperadas segun rango observado",
            "value": expected_observed,
            "status": "OK" if expected_observed == rows else "REVISAR",
            "detail": "rango observado inclusivo",
        },
    ]

    frequency_rows = [
        {"time_diff": str(diff), "count": count}
        for diff, count in sorted(diffs.items(), key=lambda item: item[0])
    ]
    null_rows = [
        {"column": column, "null_or_blank_count": null_counts[column]}
        for column in EXPECTED_COLUMNS
    ]
    numeric_rows = [
        {
            "column": column,
            "parse_errors": parse_errors[column],
            "non_positive_count": non_positive[column],
            "zero_count": zero_counts[column],
            "negative_count": negative_counts[column],
        }
        for column in NUMERIC_COLUMNS
    ]
    gap_rows = [
        {
            "previous_open_time": gap.previous_open_time,
            "next_open_time": gap.next_open_time,
            "gap_duration": gap.gap_duration,
            "missing_5m_candles": gap.missing_5m_candles,
        }
        for gap in gaps
    ]

    is_valid = all(row["status"] == "OK" for row in quality_rows)
    return {
        "csv_path": csv_path,
        "is_valid": is_valid,
        "quality_rows": quality_rows,
        "frequency_rows": frequency_rows,
        "null_rows": null_rows,
        "numeric_rows": numeric_rows,
        "gap_rows": gap_rows,
        "rows": rows,
        "min_time": min_time,
        "max_time": max_time,
        "expected_declared": expected_declared,
        "expected_observed": expected_observed,
        "duplicate_timestamps": duplicate_timestamps,
        "gaps": gaps,
        "missing_5m_candles": missing_5m_candles,
        "total_nulls": total_nulls,
        "total_parse_errors": total_parse_errors,
    }


def write_validation_tables(result: dict[str, Any], output_dir: Path) -> None:
    """Guarda las tablas CSV de control de calidad de la Fase 0."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        tables_dir / "phase0_quality_summary.csv",
        result["quality_rows"],
        ["check", "value", "status", "detail"],
    )
    write_csv(
        tables_dir / "phase0_frequency_summary.csv",
        result["frequency_rows"],
        ["time_diff", "count"],
    )
    write_csv(
        tables_dir / "phase0_null_counts.csv",
        result["null_rows"],
        ["column", "null_or_blank_count"],
    )
    write_csv(
        tables_dir / "phase0_numeric_checks.csv",
        result["numeric_rows"],
        ["column", "parse_errors", "non_positive_count", "zero_count", "negative_count"],
    )
    write_csv(
        tables_dir / "phase0_gap_report.csv",
        result["gap_rows"],
        ["previous_open_time", "next_open_time", "gap_duration", "missing_5m_candles"],
    )


def print_validation_summary(result: dict[str, Any]) -> None:
    """Muestra en consola los resultados principales de validacion."""
    print("\nFASE 0 - VALIDACION DEL DATASET BTCUSDT 5m")
    print("=" * 72)
    print(format_console_table(result["quality_rows"], ["check", "value", "status", "detail"]))
    print("\nResumen:")
    print(f"- Filas: {result['rows']:,}")
    print(f"- Rango temporal: {result['min_time']} -> {result['max_time']}")
    print(f"- Duplicados: {result['duplicate_timestamps']:,}")
    print(f"- Huecos temporales: {len(result['gaps']):,}")
    print(f"- Nulos/blancos: {result['total_nulls']:,}")
    print(f"- Errores de conversion: {result['total_parse_errors']:,}")
    print(
        "- Estado: "
        + (
            "valido"
            if result["is_valid"]
            else "requiere revision"
        )
    )
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Valida btc_5m_clean.csv para la Fase 0.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("btc_5m_clean.csv"),
        help="Ruta al CSV limpio de BTCUSDT 5m.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directorio donde guardar tablas de validacion.",
    )
    parser.add_argument(
        "--expected-start",
        type=parse_datetime,
        default=DEFAULT_EXPECTED_START,
        help="Inicio esperado del rango temporal inclusivo.",
    )
    parser.add_argument(
        "--expected-end",
        type=parse_datetime,
        default=DEFAULT_EXPECTED_END,
        help="Fin esperado del rango temporal inclusivo.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = validate_csv(args.csv, args.expected_start, args.expected_end)
    write_validation_tables(result, args.reports_dir)
    print_validation_summary(result)
    return 0 if result["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
