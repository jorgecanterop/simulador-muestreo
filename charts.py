from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from sampling_core import ManualIntervalResult, ManualSimulationResult, TheoreticalResult

COVERED_COLOR = "#1C83E1"
MISSED_COLOR = "#FF4B4B"
REFERENCE_COLOR = "#FFA421"
ESTIMATE_COLOR = "#21C354"
FIT_COLOR = "#803DF5"
SECONDARY_COLOR = "#B9789F"
ORIGINAL_INTERVAL_COLOR = "#A855F7"
EMPIRICAL_REFERENCE_COLOR = "#21C354"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _histogram_bins(size: int) -> int:
    return min(45, max(12, int(math.sqrt(size))))


def _base_layout(title: str, x_title: str, y_title: str, *, height: int = 520) -> dict:
    return {
        "title": {"text": title, "x": 0.0, "xanchor": "left"},
        "height": height,
        "margin": {"l": 55, "r": 20, "t": 85, "b": 65},
        "xaxis": {"title": x_title, "showgrid": True, "zeroline": False, "automargin": True},
        "yaxis": {"title": y_title, "showgrid": True, "zeroline": False, "automargin": True},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        "hovermode": "closest",
    }


def _add_interval_shape(
    figure: go.Figure,
    *,
    position: float,
    low: float,
    high: float,
    color: str,
    width: float,
    cap_half_width: float = 0.0,
) -> None:
    """Añade un IC como formas Plotly con color fijo.

    Las formas de ``layout.shapes`` no son recoloreadas por el tema de
    Streamlit, a diferencia de algunos trazos de línea.
    """
    figure.add_shape(
        type="line",
        x0=float(position),
        x1=float(position),
        y0=float(low),
        y1=float(high),
        xref="x",
        yref="y",
        line={"color": color, "width": width},
        layer="above",
    )

    if cap_half_width > 0:
        for endpoint in (low, high):
            figure.add_shape(
                type="line",
                x0=float(position) - cap_half_width,
                x1=float(position) + cap_half_width,
                y0=float(endpoint),
                y1=float(endpoint),
                xref="x",
                yref="y",
                line={"color": color, "width": width},
                layer="above",
            )


def interval_figure(
    estimates: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    misses: np.ndarray,
    reference: float,
    title: str,
    y_title: str,
    *,
    original_interval: ManualIntervalResult | None = None,
    empirical_reference: float | None = None,
    empirical_reference_label: str = "Centro empírico de las estimaciones",
) -> go.Figure:
    """Grafica los IC mediante formas con colores no modificables por el tema.

    Azul: contiene la referencia.
    Rojo: no contiene la referencia.
    Violeta: intervalo de la muestra original.
    """
    estimates = np.asarray(estimates, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    misses = np.asarray(misses, dtype=bool)

    repetition_numbers = np.arange(1, estimates.size + 1)

    if original_interval is not None:
        original_position = estimates.size // 2 + 1
        simulation_positions = repetition_numbers.copy()
        simulation_positions[simulation_positions >= original_position] += 1
    else:
        original_position = None
        simulation_positions = repetition_numbers

    figure = go.Figure()

    # Los intervalos se dibujan uno por uno como shapes para impedir que
    # Streamlit sustituya sus colores por el color primario del tema.
    for index, position in enumerate(simulation_positions):
        color = MISSED_COLOR if misses[index] else COVERED_COLOR
        width = 1.8 if misses[index] else 1.15
        _add_interval_shape(
            figure,
            position=float(position),
            low=float(lows[index]),
            high=float(highs[index]),
            color=color,
            width=width,
        )

    # Los puntos quedan pequeños: la información principal es el IC coloreado.
    for mask, color, label in (
        (~misses, COVERED_COLOR, "Contiene el valor de referencia"),
        (misses, MISSED_COLOR, "No contiene el valor de referencia"),
    ):
        selected = np.flatnonzero(mask)
        if selected.size == 0:
            continue

        figure.add_trace(
            go.Scatter(
                x=simulation_positions[selected],
                y=estimates[selected],
                mode="markers",
                marker={
                    "color": color,
                    "size": 4.5,
                    "line": {"width": 0},
                },
                name=label,
                customdata=np.column_stack(
                    (
                        lows[selected],
                        highs[selected],
                        repetition_numbers[selected],
                    )
                ),
                hovertemplate=(
                    "Repetición %{customdata[2]:.0f}<br>"
                    "Estimación: %{y:.5g}<br>"
                    "IC: [%{customdata[0]:.5g}, %{customdata[1]:.5g}]"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=reference,
        line={"color": REFERENCE_COLOR, "width": 2, "dash": "dash"},
    )

    if empirical_reference is not None and np.isfinite(empirical_reference):
        figure.add_hline(
            y=float(empirical_reference),
            line={"color": EMPIRICAL_REFERENCE_COLOR, "width": 2, "dash": "dot"},
        )

    # Entradas invisibles para una leyenda clara de las líneas horizontales.
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": REFERENCE_COLOR, "width": 2, "dash": "dash"},
            name="Valor de referencia",
            hoverinfo="skip",
        )
    )
    if empirical_reference is not None and np.isfinite(empirical_reference):
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={
                    "color": EMPIRICAL_REFERENCE_COLOR,
                    "width": 2,
                    "dash": "dot",
                },
                name=empirical_reference_label,
                hoverinfo="skip",
            )
        )

    if original_interval is not None and original_position is not None:
        original_estimate = (
            original_interval.sample_mean
            if original_interval.target == "Media"
            else original_interval.sample_variance
        )
        contains_reference = original_interval.low <= reference <= original_interval.high
        contains_empirical = (
            empirical_reference is not None
            and np.isfinite(empirical_reference)
            and original_interval.low <= empirical_reference <= original_interval.high
        )

        _add_interval_shape(
            figure,
            position=float(original_position),
            low=float(original_interval.low),
            high=float(original_interval.high),
            color=ORIGINAL_INTERVAL_COLOR,
            width=3.2,
            cap_half_width=0.32,
        )

        figure.add_trace(
            go.Scatter(
                x=[original_position],
                y=[original_estimate],
                mode="markers",
                marker={
                    "color": ORIGINAL_INTERVAL_COLOR,
                    "size": 9,
                    "symbol": "diamond",
                    "line": {"width": 1, "color": "white"},
                },
                name="IC de la muestra original",
                customdata=[[
                    original_interval.low,
                    original_interval.high,
                    original_interval.n,
                    "Sí" if contains_reference else "No",
                    "Sí" if contains_empirical else "No",
                ]],
                hovertemplate=(
                    "<b>Muestra original</b><br>"
                    "Estimación: %{y:.5g}<br>"
                    "IC: [%{customdata[0]:.5g}, %{customdata[1]:.5g}]<br>"
                    "n original: %{customdata[2]}<br>"
                    "¿Incluye la referencia?: %{customdata[3]}<br>"
                    "¿Incluye el centro empírico?: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    x_title = (
        "Número de repetición y muestra original"
        if original_interval is not None
        else "Número de repetición"
    )
    figure.update_layout(**_base_layout(title, x_title, y_title))

    if original_interval is not None and original_position is not None:
        tick_count = min(6, estimates.size)
        repetition_ticks = np.unique(
            np.linspace(1, estimates.size, tick_count, dtype=int)
        )
        repetition_tick_positions = repetition_ticks.copy()
        repetition_tick_positions[
            repetition_tick_positions >= original_position
        ] += 1

        tick_pairs = [
            (int(position), str(repetition))
            for position, repetition in zip(
                repetition_tick_positions,
                repetition_ticks,
            )
        ]
        tick_pairs.append((int(original_position), "Muestra original"))
        tick_pairs.sort(key=lambda pair: pair[0])

        figure.update_xaxes(
            range=[0, estimates.size + 2],
            tickmode="array",
            tickvals=[pair[0] for pair in tick_pairs],
            ticktext=[pair[1] for pair in tick_pairs],
        )
    else:
        figure.update_xaxes(range=[0, estimates.size + 1])

    return figure


def mean_histogram_figure(result: TheoreticalResult) -> go.Figure:
    values = result.means
    observed_mean = float(values.mean())
    observed_sd = float(values.std(ddof=1))
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=_histogram_bins(values.size),
            histnorm="probability density",
            name="Medias muestrales",
            opacity=0.65,
            marker={"color": COVERED_COLOR},
            hovertemplate="Media: %{x:.5g}<br>Densidad: %{y:.5g}<extra></extra>",
        )
    )
    if observed_sd > 0:
        grid = np.linspace(float(values.min()), float(values.max()), 500)
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=stats.norm.pdf(grid, observed_mean, observed_sd),
                mode="lines",
                name="Normal ajustada",
                line={"color": FIT_COLOR, "width": 3},
            )
        )
    figure.add_vline(
        x=result.true_mean,
        line={"color": REFERENCE_COLOR, "width": 2, "dash": "dash"},
        annotation_text="Media verdadera",
        annotation_position="top left",
    )
    figure.add_vline(
        x=observed_mean,
        line={"color": ESTIMATE_COLOR, "width": 2},
        annotation_text="Media de las medias",
        annotation_position="top right",
    )
    figure.update_layout(**_base_layout("Distribución de las medias muestrales", "Media muestral", "Densidad"))
    figure.update_layout(barmode="overlay")
    return figure


def variance_histogram_figure(result: TheoreticalResult) -> go.Figure:
    values = result.variances
    observed_variance = float(values.mean())
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=_histogram_bins(values.size),
            histnorm="probability density",
            name="Varianzas muestrales",
            opacity=0.65,
            marker={"color": SECONDARY_COLOR},
            hovertemplate="S²: %{x:.5g}<br>Densidad: %{y:.5g}<extra></extra>",
        )
    )
    if observed_variance > 0:
        upper = max(float(np.quantile(values, 0.995)), result.true_variance, observed_variance) * 1.15
        grid = np.linspace(0, upper, 500)
        scale = observed_variance / (result.sample_size - 1)
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=stats.chi2.pdf(grid / scale, result.sample_size - 1) / scale,
                mode="lines",
                name="χ² escalada ajustada",
                line={"color": FIT_COLOR, "width": 3},
            )
        )
    figure.add_vline(
        x=result.true_variance,
        line={"color": REFERENCE_COLOR, "width": 2, "dash": "dash"},
        annotation_text="Varianza verdadera",
        annotation_position="top left",
    )
    figure.add_vline(
        x=observed_variance,
        line={"color": ESTIMATE_COLOR, "width": 2},
        annotation_text="Media de S²",
        annotation_position="top right",
    )
    figure.update_layout(**_base_layout("Distribución de las varianzas muestrales", "Varianza muestral S²", "Densidad"))
    figure.update_layout(barmode="overlay")
    return figure


def pivot_histogram_figure(result: ManualSimulationResult) -> go.Figure:
    values = result.pivots[np.isfinite(result.pivots)]
    alpha = 1 - result.confidence
    degrees = result.degrees_freedom
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=_histogram_bins(values.size),
            histnorm="probability density",
            name="Estadístico simulado",
            opacity=0.65,
            marker={"color": COVERED_COLOR},
        )
    )

    if result.pivot_type == "chi2":
        upper = max(float(np.quantile(values, 0.995)) * 1.10, float(stats.chi2.ppf(0.995, degrees)))
        grid = np.linspace(0, upper, 500)
        density = stats.chi2.pdf(grid, degrees)
        label = f"χ² (gl={degrees})"
        low_critical, high_critical = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], degrees)
        x_title = "Q = (n−1)S²/σ²"
    elif result.pivot_type == "z":
        low, high = np.quantile(values, [0.005, 0.995])
        span = max(high - low, 1.0)
        grid = np.linspace(low - 0.12 * span, high + 0.12 * span, 500)
        density = stats.norm.pdf(grid)
        label = "Normal estándar"
        high_critical = stats.norm.ppf(1 - alpha / 2)
        low_critical = -high_critical
        x_title = "Z = (X̄ − μ)/(σ/√n)"
    else:
        low, high = np.quantile(values, [0.005, 0.995])
        span = max(high - low, 1.0)
        grid = np.linspace(low - 0.12 * span, high + 0.12 * span, 500)
        density = stats.t.pdf(grid, degrees)
        label = f"t de Student (gl={degrees})"
        high_critical = stats.t.ppf(1 - alpha / 2, degrees)
        low_critical = -high_critical
        x_title = "T = (X̄ − μ)/(S/√n)"

    figure.add_trace(
        go.Scatter(
            x=grid,
            y=density,
            mode="lines",
            name=label,
            line={"color": FIT_COLOR, "width": 3},
        )
    )
    figure.add_vline(x=low_critical, line={"color": MISSED_COLOR, "width": 2, "dash": "dash"})
    figure.add_vline(x=high_critical, line={"color": MISSED_COLOR, "width": 2, "dash": "dash"})
    figure.update_layout(**_base_layout("Distribución del estadístico estandarizado", x_title, "Densidad"))
    figure.update_layout(barmode="overlay")
    return figure
