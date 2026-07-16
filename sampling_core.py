from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy import stats

MAX_REPETITIONS = 2_000
MAX_SAMPLE_SIZE = 100_000
MAX_TOTAL_DRAWS = 25_000_000
MAX_VALUES_PER_BATCH = 1_500_000


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    default: float | int
    integer: bool = False
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int = 0.1


DISTRIBUTION_SPECS: dict[str, tuple[ParameterSpec, ...]] = {
    "Bernoulli": (
        ParameterSpec("p", "Probabilidad p", 0.50, minimum=0.0, maximum=1.0, step=0.01),
    ),
    "Binomial": (
        ParameterSpec("m", "Número de ensayos", 10, integer=True, minimum=1, step=1),
        ParameterSpec("p", "Probabilidad p", 0.50, minimum=0.0, maximum=1.0, step=0.01),
    ),
    "Poisson": (
        ParameterSpec("lambda", "Tasa λ", 4.00, minimum=0.0001, step=0.10),
    ),
    "Binomial negativa": (
        ParameterSpec("r", "Número de éxitos r", 5, integer=True, minimum=1, step=1),
        ParameterSpec("p", "Probabilidad p", 0.40, minimum=0.0001, maximum=1.0, step=0.01),
    ),
    "Geométrica": (
        ParameterSpec("p", "Probabilidad p", 0.35, minimum=0.0001, maximum=1.0, step=0.01),
    ),
    "Hipergeométrica": (
        ParameterSpec("N", "Tamaño poblacional N", 40, integer=True, minimum=1, step=1),
        ParameterSpec("K", "Éxitos en la población K", 12, integer=True, minimum=0, step=1),
        ParameterSpec("m", "Extracciones", 8, integer=True, minimum=0, step=1),
    ),
    "Normal": (
        ParameterSpec("mu", "Media μ", 0.00, step=0.10),
        ParameterSpec("sigma", "Desviación estándar σ", 1.00, minimum=0.0001, step=0.10),
    ),
    "Exponencial": (
        ParameterSpec("lambda", "Tasa λ", 1.00, minimum=0.0001, step=0.10),
    ),
    "Uniforme": (
        ParameterSpec("a", "Límite inferior a", 0.00, step=0.10),
        ParameterSpec("b", "Límite superior b", 10.00, step=0.10),
    ),
    "Gamma": (
        ParameterSpec("alpha", "Forma α", 2.00, minimum=0.0001, step=0.10),
        ParameterSpec("theta", "Escala θ", 2.00, minimum=0.0001, step=0.10),
    ),
    "Chi-cuadrado": (
        ParameterSpec("nu", "Grados de libertad ν", 5.00, minimum=0.0001, step=0.10),
    ),
    "t de Student": (
        ParameterSpec("nu", "Grados de libertad ν", 8.00, minimum=0.0001, step=0.10),
    ),
    "F de Fisher": (
        ParameterSpec("d1", "Grados de libertad ν₁", 5.00, minimum=0.0001, step=0.10),
        ParameterSpec("d2", "Grados de libertad ν₂", 12.00, minimum=0.0001, step=0.10),
    ),
}


@dataclass
class TheoreticalResult:
    distribution_name: str
    sample_size: int
    repetitions: int
    confidence: float
    true_mean: float
    true_variance: float
    means: np.ndarray
    variances: np.ndarray
    mean_lows: np.ndarray
    mean_highs: np.ndarray
    variance_lows: np.ndarray
    variance_highs: np.ndarray
    mean_misses: np.ndarray
    variance_misses: np.ndarray
    mean_interval_name: str
    warning: str


@dataclass
class ManualIntervalResult:
    target: str
    confidence: float
    n: int
    sample_mean: float
    sample_variance: float
    standard_error: float | None
    degrees_freedom: int
    method: str
    low: float
    high: float
    warning: str


@dataclass
class ManualSimulationResult:
    target: str
    simulation_method: str
    interval_name: str
    sample_size: int
    repetitions: int
    confidence: float
    reference: float
    estimates: np.ndarray
    lows: np.ndarray
    highs: np.ndarray
    misses: np.ndarray
    pivots: np.ndarray
    pivot_type: str
    degrees_freedom: int
    warning: str


def _finite_number(value: float | int | str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} debe ser numérico.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} debe ser finito.")
    return parsed


def _positive(value: float | int | str, label: str) -> float:
    parsed = _finite_number(value, label)
    if parsed <= 0:
        raise ValueError(f"{label} debe ser mayor que 0.")
    return parsed


def _probability(value: float | int | str, label: str = "p", *, allow_zero: bool = True) -> float:
    parsed = _finite_number(value, label)
    valid = 0 <= parsed <= 1 if allow_zero else 0 < parsed <= 1
    if not valid:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{label} debe pertenecer a {interval}.")
    return parsed


def _integer(value: float | int | str, label: str, *, minimum: int = 0) -> int:
    parsed = _finite_number(value, label)
    rounded = int(round(parsed))
    if not math.isclose(parsed, rounded):
        raise ValueError(f"{label} debe ser entero.")
    if rounded < minimum:
        raise ValueError(f"{label} debe ser mayor o igual que {minimum}.")
    return rounded


def validate_simulation_size(sample_size: int, repetitions: int) -> None:
    if sample_size < 2:
        raise ValueError("El tamaño muestral debe ser al menos 2.")
    if repetitions < 10:
        raise ValueError("El número de repeticiones debe ser al menos 10.")
    if sample_size > MAX_SAMPLE_SIZE:
        raise ValueError(f"El tamaño muestral máximo es {MAX_SAMPLE_SIZE:,}.")
    if repetitions > MAX_REPETITIONS:
        raise ValueError(f"El máximo de repeticiones es {MAX_REPETITIONS:,}.")
    if sample_size * repetitions > MAX_TOTAL_DRAWS:
        raise ValueError(
            "La combinación de tamaño muestral y repeticiones es demasiado grande. "
            f"El máximo permitido es {MAX_TOTAL_DRAWS:,} observaciones simuladas."
        )


def validate_confidence(confidence_percent: float) -> float:
    confidence = _finite_number(confidence_percent, "Nivel de confianza") / 100
    if not 0.50 <= confidence < 1:
        raise ValueError("El nivel de confianza debe estar entre 50% y menos de 100%.")
    return confidence


def build_distribution(name: str, parameters: Mapping[str, float | int]):
    if name == "Bernoulli":
        return stats.bernoulli(_probability(parameters["p"]))
    if name == "Binomial":
        return stats.binom(_integer(parameters["m"], "Número de ensayos", minimum=1), _probability(parameters["p"]))
    if name == "Poisson":
        return stats.poisson(_positive(parameters["lambda"], "λ"))
    if name == "Binomial negativa":
        return stats.nbinom(
            _integer(parameters["r"], "Número de éxitos r", minimum=1),
            _probability(parameters["p"], allow_zero=False),
        )
    if name == "Geométrica":
        return stats.geom(_probability(parameters["p"], allow_zero=False))
    if name == "Hipergeométrica":
        population = _integer(parameters["N"], "N", minimum=1)
        successes = _integer(parameters["K"], "K", minimum=0)
        draws = _integer(parameters["m"], "Extracciones", minimum=0)
        if successes > population or draws > population:
            raise ValueError("Debe cumplirse K ≤ N y extracciones ≤ N.")
        return stats.hypergeom(population, successes, draws)
    if name == "Normal":
        return stats.norm(_finite_number(parameters["mu"], "μ"), _positive(parameters["sigma"], "σ"))
    if name == "Exponencial":
        return stats.expon(scale=1 / _positive(parameters["lambda"], "λ"))
    if name == "Uniforme":
        lower = _finite_number(parameters["a"], "a")
        upper = _finite_number(parameters["b"], "b")
        if upper <= lower:
            raise ValueError("Debe cumplirse b > a.")
        return stats.uniform(lower, upper - lower)
    if name == "Gamma":
        return stats.gamma(_positive(parameters["alpha"], "α"), scale=_positive(parameters["theta"], "θ"))
    if name == "Chi-cuadrado":
        return stats.chi2(_positive(parameters["nu"], "ν"))
    if name == "t de Student":
        return stats.t(_positive(parameters["nu"], "ν"))
    if name == "F de Fisher":
        return stats.f(_positive(parameters["d1"], "ν₁"), _positive(parameters["d2"], "ν₂"))
    raise ValueError("Distribución no reconocida.")


def sample_statistics(
    draw: Callable[[tuple[int, int]], np.ndarray],
    repetitions: int,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty(repetitions, dtype=float)
    variances = np.empty(repetitions, dtype=float)
    batch_size = max(1, min(repetitions, MAX_VALUES_PER_BATCH // sample_size))

    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        expected_shape = (stop - start, sample_size)
        sample = np.asarray(draw(expected_shape), dtype=float).reshape(expected_shape)
        means[start:stop] = sample.mean(axis=1)
        variances[start:stop] = sample.var(axis=1, ddof=1)
    return means, variances


def mean_intervals(
    means: np.ndarray,
    variances: np.ndarray,
    sample_size: int,
    alpha: float,
    population_variance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if population_variance is not None:
        critical = stats.norm.ppf(1 - alpha / 2)
        half_width = critical * math.sqrt(population_variance / sample_size)
        return means - half_width, means + half_width, "IC Z (varianza poblacional conocida)"

    critical = stats.t.ppf(1 - alpha / 2, sample_size - 1)
    half_width = critical * np.sqrt(variances / sample_size)
    return means - half_width, means + half_width, "IC t (varianza estimada)"


def variance_intervals(variances: np.ndarray, sample_size: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    degrees = sample_size - 1
    lower_q, upper_q = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], degrees)
    return degrees * variances / upper_q, degrees * variances / lower_q


def simulate_theoretical(
    distribution_name: str,
    parameters: Mapping[str, float | int],
    sample_size: int,
    repetitions: int,
    confidence_percent: float,
    seed: int,
    mean_method: str,
) -> TheoreticalResult:
    validate_simulation_size(sample_size, repetitions)
    confidence = validate_confidence(confidence_percent)
    alpha = 1 - confidence
    distribution = build_distribution(distribution_name, parameters)

    true_mean = float(distribution.mean())
    true_variance = float(distribution.var())
    if not np.isfinite(true_mean):
        raise ValueError("La distribución seleccionada no posee media finita con esos parámetros.")
    if not np.isfinite(true_variance) or true_variance < 0:
        raise ValueError("La distribución seleccionada no posee varianza finita con esos parámetros.")

    rng = np.random.default_rng(seed)
    means, variances = sample_statistics(
        lambda shape: distribution.rvs(size=shape, random_state=rng), repetitions, sample_size
    )
    population_variance = true_variance if mean_method == "z" else None
    mean_lows, mean_highs, mean_interval_name = mean_intervals(
        means, variances, sample_size, alpha, population_variance
    )
    variance_lows, variance_highs = variance_intervals(variances, sample_size, alpha)
    mean_misses = (mean_lows > true_mean) | (mean_highs < true_mean)
    variance_misses = (variance_lows > true_variance) | (variance_highs < true_variance)

    warning = ""
    if distribution_name != "Normal":
        warning = (
            "Los intervalos y el ajuste χ² para la varianza son exactos bajo normalidad poblacional. "
            "El intervalo t para la media también es exacto bajo normalidad y aproximado en otros casos."
        )

    return TheoreticalResult(
        distribution_name=distribution_name,
        sample_size=sample_size,
        repetitions=repetitions,
        confidence=confidence,
        true_mean=true_mean,
        true_variance=true_variance,
        means=means,
        variances=variances,
        mean_lows=mean_lows,
        mean_highs=mean_highs,
        variance_lows=variance_lows,
        variance_highs=variance_highs,
        mean_misses=mean_misses,
        variance_misses=variance_misses,
        mean_interval_name=mean_interval_name,
        warning=warning,
    )


def clean_manual_data(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError("Ingrese al menos dos observaciones numéricas.")
    return array


def calculate_manual_interval(
    values,
    target: str,
    confidence_percent: float,
    known_population_variance: float | None,
) -> ManualIntervalResult:
    data = clean_manual_data(values)
    confidence = validate_confidence(confidence_percent)
    alpha = 1 - confidence
    n = int(data.size)
    degrees = n - 1
    sample_mean = float(data.mean())
    sample_variance = float(data.var(ddof=1))

    if known_population_variance is not None:
        known_population_variance = _positive(known_population_variance, "Varianza poblacional")

    if target == "Media":
        if known_population_variance is not None:
            standard_error = math.sqrt(known_population_variance / n)
            critical = stats.norm.ppf(1 - alpha / 2)
            method = "Intervalo Z para μ (σ² conocida)"
        else:
            standard_error = math.sqrt(sample_variance / n)
            critical = stats.t.ppf(1 - alpha / 2, degrees)
            method = f"Intervalo t para μ (gl = {degrees})"
        low = sample_mean - critical * standard_error
        high = sample_mean + critical * standard_error
        warning = ""
    elif target == "Varianza":
        low_arr, high_arr = variance_intervals(np.asarray([sample_variance]), n, alpha)
        low, high = float(low_arr[0]), float(high_arr[0])
        standard_error = None
        method = "Intervalo χ² para σ²"
        warning = "El intervalo χ² para la varianza supone una población normal."
    else:
        raise ValueError("Parámetro objetivo no reconocido.")

    return ManualIntervalResult(
        target=target,
        confidence=confidence,
        n=n,
        sample_mean=sample_mean,
        sample_variance=sample_variance,
        standard_error=standard_error,
        degrees_freedom=degrees,
        method=method,
        low=low,
        high=high,
        warning=warning,
    )


def simulate_from_manual_data(
    values,
    target: str,
    confidence_percent: float,
    known_population_variance: float | None,
    simulation_method: str,
    sample_size: int,
    repetitions: int,
    seed: int,
) -> ManualSimulationResult:
    data = clean_manual_data(values)
    validate_simulation_size(sample_size, repetitions)
    confidence = validate_confidence(confidence_percent)
    alpha = 1 - confidence

    mean_reference = float(data.mean())
    sample_variance = float(data.var(ddof=1))
    if known_population_variance is not None:
        known_population_variance = _positive(known_population_variance, "Varianza poblacional")
    variance_reference = known_population_variance if known_population_variance is not None else sample_variance
    if variance_reference <= 0:
        raise ValueError("La muestra no presenta variabilidad suficiente para simular.")

    rng = np.random.default_rng(seed)
    if simulation_method == "Normal ajustada":
        draw = lambda shape: rng.normal(mean_reference, math.sqrt(variance_reference), size=shape)
    elif simulation_method == "Bootstrap":
        draw = lambda shape: rng.choice(data, size=shape, replace=True)
    else:
        raise ValueError("Método de simulación no reconocido.")

    means, variances = sample_statistics(draw, repetitions, sample_size)
    degrees = sample_size - 1

    if target == "Media":
        estimates = means
        reference = mean_reference
        if known_population_variance is not None:
            lows, highs, _ = mean_intervals(
                means, variances, sample_size, alpha, population_variance=variance_reference
            )
            pivots = (means - reference) / math.sqrt(variance_reference / sample_size)
            pivot_type = "z"
            interval_name = "IC Z para la media"
        else:
            lows, highs, _ = mean_intervals(means, variances, sample_size, alpha)
            standard_errors = np.sqrt(variances / sample_size)
            pivots = np.divide(
                means - reference,
                standard_errors,
                out=np.full(repetitions, np.nan),
                where=standard_errors > 0,
            )
            pivot_type = "t"
            interval_name = "IC t para la media"
        warning = ""
    elif target == "Varianza":
        estimates = variances
        reference = variance_reference
        lows, highs = variance_intervals(variances, sample_size, alpha)
        pivots = degrees * variances / reference
        pivot_type = "chi2"
        interval_name = "IC χ² para la varianza"
        warning = (
            "La referencia χ² y los intervalos de varianza son exactos cuando el modelo generador es normal. "
            "Con bootstrap se muestran como comparación didáctica."
        )
    else:
        raise ValueError("Parámetro objetivo no reconocido.")

    finite_pivots = np.asarray(pivots, dtype=float)
    if np.isfinite(finite_pivots).sum() < 2:
        raise ValueError("No fue posible obtener suficientes estadísticos estandarizados finitos.")

    misses = (lows > reference) | (highs < reference)
    return ManualSimulationResult(
        target=target,
        simulation_method=simulation_method,
        interval_name=interval_name,
        sample_size=sample_size,
        repetitions=repetitions,
        confidence=confidence,
        reference=reference,
        estimates=estimates,
        lows=lows,
        highs=highs,
        misses=misses,
        pivots=finite_pivots,
        pivot_type=pivot_type,
        degrees_freedom=degrees,
        warning=warning,
    )


def format_number(value: float | None, decimals: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    value = float(value)
    magnitude = abs(value)
    if magnitude and (magnitude >= 1e5 or magnitude < 1e-4):
        return f"{value:.3e}"
    return f"{value:.{decimals}f}"
