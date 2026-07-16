import numpy as np

from charts import (
    COVERED_COLOR,
    MISSED_COLOR,
    ORIGINAL_INTERVAL_COLOR,
    interval_figure,
)
from sampling_core import calculate_manual_interval


def _vertical_shapes(figure):
    return [
        shape
        for shape in figure.layout.shapes
        if shape.type == "line" and shape.x0 == shape.x1
    ]


def test_interval_shapes_have_blue_and_red_colors():
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

    colors = [shape.line.color for shape in _vertical_shapes(figure)]
    assert colors.count(COVERED_COLOR) == 3
    assert colors.count(MISSED_COLOR) == 1


def test_manual_interval_is_violet_and_centered():
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

    expected_position = estimates.size // 2 + 1
    violet_vertical = [
        shape
        for shape in _vertical_shapes(figure)
        if shape.line.color == ORIGINAL_INTERVAL_COLOR
    ]
    assert len(violet_vertical) == 1
    assert violet_vertical[0].x0 == expected_position
    assert violet_vertical[0].line.width == 3.2

    marker = next(
        trace for trace in figure.data
        if trace.name == "IC de la muestra original"
    )
    assert marker.marker.size == 9
    assert marker.marker.color == ORIGINAL_INTERVAL_COLOR
