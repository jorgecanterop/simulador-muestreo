import numpy as np

from charts import (
    COVERED_COLOR,
    MISSED_COLOR,
    ORIGINAL_INTERVAL_COLOR,
    interval_figure,
)
from sampling_core import calculate_manual_interval


def _line_traces_with_color(figure, color):
    return [
        trace
        for trace in figure.data
        if getattr(trace, "mode", None) == "lines"
        and trace.line.color == color
    ]


def test_interval_segments_use_explicit_category_colors():
    estimates = np.array([10.0, 11.0, 12.0, 13.0])
    lows = np.array([9.0, 10.0, 11.0, 12.4])
    highs = np.array([11.0, 12.0, 13.0, 13.6])
    misses = np.array([False, False, False, True])

    figure = interval_figure(
        estimates,
        lows,
        highs,
        misses,
        reference=12.0,
        title="Intervalos",
        y_title="Media",
    )

    blue_lines = _line_traces_with_color(figure, COVERED_COLOR)
    red_lines = _line_traces_with_color(figure, MISSED_COLOR)

    # Dos trazos por categoría: segmentos verticales y tapas horizontales.
    assert len(blue_lines) == 2
    assert len(red_lines) == 2

    covered_marker = next(
        trace for trace in figure.data
        if trace.name == "Contiene el valor de referencia"
    )
    missed_marker = next(
        trace for trace in figure.data
        if trace.name == "No contiene el valor de referencia"
    )
    assert covered_marker.marker.color == COVERED_COLOR
    assert missed_marker.marker.color == MISSED_COLOR


def test_original_interval_is_explicitly_violet_and_centered():
    original = calculate_manual_interval(
        values=[10, 11, 12, 13, 14],
        target="Media",
        confidence_percent=95.0,
        known_population_variance=None,
    )
    estimates = np.arange(1.0, 11.0)
    lows = estimates - 1.0
    highs = estimates + 1.0
    misses = np.zeros(estimates.size, dtype=bool)

    figure = interval_figure(
        estimates,
        lows,
        highs,
        misses,
        reference=5.5,
        title="Intervalos",
        y_title="Media",
        original_interval=original,
        empirical_reference=float(estimates.mean()),
    )

    violet_lines = _line_traces_with_color(figure, ORIGINAL_INTERVAL_COLOR)
    assert len(violet_lines) == 2
    assert all(trace.line.width == 6 for trace in violet_lines)

    original_marker = next(
        trace for trace in figure.data
        if trace.name == "IC de la muestra original"
    )
    expected_position = estimates.size // 2 + 1
    assert list(original_marker.x) == [expected_position]
    assert original_marker.marker.color == ORIGINAL_INTERVAL_COLOR
    assert original_marker.marker.symbol == "diamond"
