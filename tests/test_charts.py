import numpy as np

from charts import (
    COVERED_COLOR,
    MISSED_COLOR,
    ORIGINAL_INTERVAL_COLOR,
    interval_figure,
)
from sampling_core import calculate_manual_interval


def test_interval_lines_and_points_share_their_category_color():
    estimates = np.array([10.0, 11.0, 12.0, 13.0])
    lows = np.array([9.0, 10.0, 11.0, 11.5])
    highs = np.array([11.0, 12.0, 13.0, 12.5])
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

    covered = next(
        trace for trace in figure.data
        if trace.name == "Contiene el valor de referencia"
    )
    missed = next(
        trace for trace in figure.data
        if trace.name == "No contiene el valor de referencia"
    )

    assert covered.marker.color == COVERED_COLOR
    assert covered.error_y.color == COVERED_COLOR
    assert missed.marker.color == MISSED_COLOR
    assert missed.error_y.color == MISSED_COLOR


def test_original_interval_is_colored_and_inserted_in_the_middle():
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
        title="Intervalos de confianza de las medias",
        y_title="Media muestral",
        original_interval=original,
        empirical_reference=float(estimates.mean()),
        empirical_reference_label="Media de las medias simuladas",
    )

    original_trace = next(
        trace for trace in figure.data
        if trace.name == "IC de la muestra original"
    )

    expected_middle_position = estimates.size // 2 + 1
    assert list(original_trace.x) == [expected_middle_position]
    assert original_trace.marker.color == ORIGINAL_INTERVAL_COLOR
    assert original_trace.error_y.color == ORIGINAL_INTERVAL_COLOR
    assert original_trace.marker.symbol == "diamond"

    covered_trace = next(
        trace for trace in figure.data
        if trace.name == "Contiene el valor de referencia"
    )
    assert expected_middle_position not in list(covered_trace.x)
