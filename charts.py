from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from sampling_core import ManualSimulationResult, TheoreticalResult

COVERED_COLOR = "#1C83E1"
MISSED_COLOR = "#FF4B4B"
REFERENCE_COLOR = "#FFA421"
ESTIMATE_COLOR = "#21C354"
FIT_COLOR = "#803DF5"
SECONDARY_COLOR = "#B9789F"

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


def _segments(x: np.ndarray, lows: np.ndarray, highs: np.ndarray, mask: np.ndarray):
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return [], []
    xs: list[float | None] = []
    ys: list[float | None] = []
    for index in selected:
        xs.extend((float(x[index]), float(x[index]), None))
        ys.extend((float(lows[index]), float(highs[index]), None))
    return xs, ys


def interval_figure(
    estimates: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    misses: np.ndarray,
    reference: float,
    title: str,
    y_title: str,
) -> go.Figure:
    x = np.arange(1, estimates.size + 1)
    covered = ~misses
    figure = go.Figure()

    for mask, color, label in (
        (covered, COVERED_COLOR, "Contiene el valor de referencia"),
        (misses, MISSED_COLOR, "No contiene el valor de referencia"),
    ):
        segment_x, segment_y = _segments(x, lows, highs, mask)
        if segment_x:
            figure.add_trace(
                go.Scatter(
                    x=segment_x,
                    y=segment_y,
                    mode="lines",
                    line={"color": color, "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        selected = np.flatnonzero(mask)
        if selected.size:
            figure.add_trace(
                go.Scatter(
                    x=x[selected],
                    y=estimates[selected],
                    mode="markers",
                    marker={"color": color, "size": 7},
                    name=label,
                    customdata=np.column_stack((lows[selected], highs[selected])),
                    hovertemplate=(
                        "Repetición %{x}<br>Estimación: %{y:.5g}<br>"
                        "IC: [%{customdata[0]:.5g}, %{customdata[1]:.5g}]<extra></extra>"
                    ),
                )
            )

    figure.add_hline(
        y=reference,
        line={"color": REFERENCE_COLOR, "width": 2, "dash": "dash"},
        annotation_text="Valor de referencia",
        annotation_position="top right",
    )
    figure.update_layout(**_base_layout(title, "Número de repetición", y_title))
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
