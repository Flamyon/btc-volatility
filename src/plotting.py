"""Graficos SVG ligeros sin dependencias externas."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import html
import math


Color = str


def write_line_svg(
    path: Path,
    times: Sequence[str],
    values: Sequence[float],
    title: str,
    y_label: str,
    color: Color = "#1f77b4",
    width: int = 1100,
    height: int = 430,
    max_points: int = 4500,
) -> None:
    """Escribe un grafico de linea SVG con ejes y rejilla."""
    points = _downsample_minmax(times, values, max_points=max_points)
    sampled_times = [time for time, _ in points]
    sampled_values = [value for _, value in points]
    svg = _line_svg_document(
        times=sampled_times,
        values=sampled_values,
        title=title,
        y_label=y_label,
        color=color,
        width=width,
        height=height,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_two_panel_svg(
    path: Path,
    times: Sequence[str],
    first_values: Sequence[float],
    second_values: Sequence[float],
    title: str,
    first_y_label: str,
    second_y_label: str,
    first_color: Color = "#2a6fbb",
    second_color: Color = "#b45f06",
    width: int = 1100,
    height: int = 620,
    max_points: int = 4000,
) -> None:
    """Escribe dos series en paneles verticales con ejes independientes."""
    first_points = _downsample_minmax(times, first_values, max_points=max_points)
    second_points = _downsample_minmax(times, second_values, max_points=max_points)

    margin = {"left": 88, "right": 34, "top": 58, "bottom": 54}
    gap = 48
    panel_height = (height - margin["top"] - margin["bottom"] - gap) / 2
    plot_width = width - margin["left"] - margin["right"]
    first_top = margin["top"]
    second_top = margin["top"] + panel_height + gap

    elements = [
        _svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
            f"{_esc(title)}</text>"
        ),
    ]
    elements.extend(
        _panel_elements(
            first_points,
            first_top,
            panel_height,
            margin,
            plot_width,
            first_y_label,
            first_color,
            include_x_labels=False,
        )
    )
    elements.extend(
        _panel_elements(
            second_points,
            second_top,
            panel_height,
            margin,
            plot_width,
            second_y_label,
            second_color,
            include_x_labels=True,
        )
    )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def _line_svg_document(
    times: Sequence[str],
    values: Sequence[float],
    title: str,
    y_label: str,
    color: Color,
    width: int,
    height: int,
) -> str:
    margin = {"left": 88, "right": 34, "top": 58, "bottom": 58}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    elements = [
        _svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="18" font-weight="700">'
            f"{_esc(title)}</text>"
        ),
    ]
    elements.extend(
        _panel_elements(
            list(zip(times, values)),
            margin["top"],
            plot_height,
            margin,
            plot_width,
            y_label,
            color,
            include_x_labels=True,
        )
    )
    elements.append("</svg>")
    return "\n".join(elements)


def _panel_elements(
    points: Sequence[tuple[str, float]],
    top: float,
    panel_height: float,
    margin: dict[str, int],
    plot_width: float,
    y_label: str,
    color: Color,
    include_x_labels: bool,
) -> list[str]:
    values = [value for _, value in points]
    y_min, y_max = _expanded_range(min(values), max(values))
    n = len(points)

    def x_coord(index: int) -> float:
        if n <= 1:
            return margin["left"]
        return margin["left"] + plot_width * index / (n - 1)

    def y_coord(value: float) -> float:
        return top + panel_height - panel_height * (value - y_min) / (y_max - y_min)

    elements: list[str] = []
    y_ticks = _nice_ticks(y_min, y_max, count=5)
    for tick in y_ticks:
        y = y_coord(tick)
        elements.append(
            f'<line x1="{margin["left"]}" y1="{y:.2f}" '
            f'x2="{margin["left"] + plot_width:.2f}" y2="{y:.2f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#333333">'
            f"{_format_tick(tick)}</text>"
        )

    axis_bottom = top + panel_height
    elements.append(
        f'<rect x="{margin["left"]}" y="{top:.2f}" width="{plot_width:.2f}" '
        f'height="{panel_height:.2f}" fill="none" stroke="#222222" stroke-width="1"/>'
    )

    polyline = " ".join(
        f"{x_coord(index):.2f},{y_coord(value):.2f}"
        for index, (_, value) in enumerate(points)
        if math.isfinite(value)
    )
    elements.append(
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="1.25" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    elements.append(
        (
            f'<text transform="translate(18,{top + panel_height / 2:.1f}) rotate(-90)" '
            f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13" '
            f'fill="#222222">{_esc(y_label)}</text>'
        )
    )

    if include_x_labels and points:
        label_indices = [0, len(points) // 2, len(points) - 1]
        for index in label_indices:
            label = _short_date(points[index][0])
            x = x_coord(index)
            elements.append(
                f'<line x1="{x:.2f}" y1="{axis_bottom:.2f}" x2="{x:.2f}" '
                f'y2="{axis_bottom + 5:.2f}" stroke="#222222" stroke-width="1"/>'
            )
            elements.append(
                f'<text x="{x:.2f}" y="{axis_bottom + 22:.2f}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11" fill="#333333">'
                f"{_esc(label)}</text>"
            )
        elements.append(
            f'<text x="{margin["left"] + plot_width / 2:.2f}" y="{axis_bottom + 43:.2f}" '
            f'text-anchor="middle" font-family="Arial, sans-serif" font-size="13" '
            f'fill="#222222">Tiempo</text>'
        )

    return elements


def _downsample_minmax(
    times: Sequence[str],
    values: Sequence[float],
    max_points: int,
) -> list[tuple[str, float]]:
    """Reduce puntos conservando minimos y maximos por bloque temporal."""
    clean = [(time, value) for time, value in zip(times, values) if math.isfinite(value)]
    if len(clean) <= max_points:
        return clean

    bucket_size = max(1, math.ceil(len(clean) / max(2, max_points // 2)))
    selected: list[tuple[int, str, float]] = []

    for start in range(0, len(clean), bucket_size):
        bucket = clean[start : start + bucket_size]
        if not bucket:
            continue
        min_offset, (_, min_value) = min(enumerate(bucket), key=lambda item: item[1][1])
        max_offset, (_, max_value) = max(enumerate(bucket), key=lambda item: item[1][1])
        for offset in sorted({min_offset, max_offset}):
            time, value = bucket[offset]
            selected.append((start + offset, time, value))

    selected.sort(key=lambda item: item[0])
    return [(time, value) for _, time, value in selected]


def _expanded_range(y_min: float, y_max: float) -> tuple[float, float]:
    if not math.isfinite(y_min) or not math.isfinite(y_max):
        return 0.0, 1.0
    if y_min == y_max:
        delta = abs(y_min) * 0.05 or 1.0
        return y_min - delta, y_max + delta
    padding = (y_max - y_min) * 0.05
    return y_min - padding, y_max + padding


def _nice_ticks(y_min: float, y_max: float, count: int) -> list[float]:
    if count <= 1:
        return [y_min]
    return [y_min + (y_max - y_min) * index / (count - 1) for index in range(count)]


def _format_tick(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    if abs_value >= 1:
        return f"{value:.2f}"
    if abs_value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.2e}"


def _short_date(timestamp: str) -> str:
    return timestamp[:10]


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )
