import numpy as np

from charts import interval_figure
from sampling_core import calculate_manual_interval


def test_original_interval_is_added_to_interval_plot():
    original = calculate_manual_interval(
        values=[10, 11, 12, 13, 14],
        target="Media",
        confidence_percent=95.0,
        known_population_variance=None,
    )
    estimates = np.array([11.0, 12.0, 13.0])
    lows = np.array([10.0, 11.0, 12.0])
    highs = np.array([12.0, 13.0, 14.0])
    misses = np.array([False, False, True])

    figure = interval_figure(
        estimates,
        lows,
        highs,
        misses,
        reference=12.0,
        title="Intervalos de confianza de las medias",
        y_title="Media muestral",
        original_interval=original,
        empirical_reference=float(estimates.mean()),
        empirical_reference_label="Media de las medias simuladas",
    )

    trace_names = [trace.name for trace in figure.data if trace.name]
    assert "IC de la muestra original" in trace_names
    original_trace = next(
        trace for trace in figure.data if trace.name == "IC de la muestra original"
    )
    assert list(original_trace.x) == [0]
    assert original_trace.marker.symbol == "diamond"
