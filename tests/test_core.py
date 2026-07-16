import numpy as np

from sampling_core import (
    build_distribution,
    calculate_manual_interval,
    simulate_from_manual_data,
    simulate_theoretical,
)


def test_normal_distribution_parameters():
    distribution = build_distribution("Normal", {"mu": 3.0, "sigma": 2.0})
    assert distribution.mean() == 3.0
    assert distribution.var() == 4.0


def test_theoretical_simulation_shapes():
    result = simulate_theoretical(
        distribution_name="Normal",
        parameters={"mu": 0.0, "sigma": 1.0},
        sample_size=20,
        repetitions=100,
        confidence_percent=95.0,
        seed=123,
        mean_method="z",
    )
    assert result.means.shape == (100,)
    assert result.variances.shape == (100,)
    assert result.mean_lows.shape == (100,)
    assert 0 <= result.mean_misses.mean() <= 1


def test_manual_mean_interval_is_ordered():
    result = calculate_manual_interval(
        values=[1, 2, 3, 4, 5],
        target="Media",
        confidence_percent=95.0,
        known_population_variance=None,
    )
    assert result.low < result.sample_mean < result.high


def test_manual_bootstrap_simulation():
    result = simulate_from_manual_data(
        values=np.arange(1, 11),
        target="Media",
        confidence_percent=95.0,
        known_population_variance=None,
        simulation_method="Bootstrap",
        sample_size=10,
        repetitions=100,
        seed=123,
    )
    assert result.estimates.shape == (100,)
    assert result.pivots.shape == (100,)
