from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from charts import (
    PLOTLY_CONFIG,
    interval_figure,
    mean_histogram_figure,
    pivot_histogram_figure,
    variance_histogram_figure,
)
from sampling_core import (
    DISTRIBUTION_SPECS,
    MAX_REPETITIONS,
    MAX_SAMPLE_SIZE,
    MAX_TOTAL_DRAWS,
    ManualIntervalResult,
    ManualSimulationResult,
    ParameterSpec,
    TheoreticalResult,
    calculate_manual_interval,
    format_number,
    simulate_from_manual_data,
    simulate_theoretical,
)

st.set_page_config(
    page_title="Simulador de muestreo e intervalos de confianza",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Solo se ajustan la geometría y el comportamiento responsive. Los colores,
# la tipografía y el tamaño base permanecen bajo el control del tema activo
# de Streamlit y de las preferencias del navegador.
st.markdown(
    """
    <style>
      .block-container {
        width: 100%;
        max-width: 1120px;
        margin: 0 auto;
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
      }

      div[data-testid="stButton"] > button {
        min-height: 2.75rem;
      }

      div[data-testid="stPills"] [role="listbox"],
      div[data-testid="stSegmentedControl"] {
        flex-wrap: wrap;
      }

      @media (max-width: 640px) {
        .block-container {
          max-width: none;
          padding-left: .75rem;
          padding-right: .75rem;
          padding-top: .7rem;
        }

        div[data-testid="stButton"] > button {
          width: 100%;
        }

        div[data-testid="stPlotlyChart"] {
          margin-bottom: 1.25rem;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


DISTRIBUTION_GROUPS = {
    "Discretas": (
        "Bernoulli",
        "Binomial",
        "Poisson",
        "Binomial negativa",
        "Geométrica",
        "Hipergeométrica",
    ),
    "Continuas": (
        "Normal",
        "Exponencial",
        "Uniforme",
        "Gamma",
        "Chi-cuadrado",
        "t de Student",
        "F de Fisher",
    ),
}


def _init_state() -> None:
    defaults = {
        "theoretical_result": None,
        "manual_interval_result": None,
        "manual_simulation_result": None,
        "manual_frame": pd.DataFrame({"Observación": pd.Series([np.nan] * 8, dtype="float64")}),
        "editor_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_manual_results() -> None:
    st.session_state.manual_interval_result = None
    st.session_state.manual_simulation_result = None


def _replace_manual_data(values: list[float] | None = None) -> None:
    values = values or [np.nan] * 8
    st.session_state.manual_frame = pd.DataFrame(
        {"Observación": pd.Series(values, dtype="float64")}
    )
    st.session_state.editor_version += 1
    _clear_manual_results()


def _number_input_for_parameter(distribution: str, spec: ParameterSpec):
    key = f"parameter_{distribution}_{spec.key}"
    if spec.integer:
        return st.number_input(
            spec.label,
            min_value=int(spec.minimum) if spec.minimum is not None else None,
            max_value=int(spec.maximum) if spec.maximum is not None else None,
            value=int(spec.default),
            step=int(spec.step),
            format="%d",
            key=key,
        )
    return st.number_input(
        spec.label,
        min_value=float(spec.minimum) if spec.minimum is not None else None,
        max_value=float(spec.maximum) if spec.maximum is not None else None,
        value=float(spec.default),
        step=float(spec.step),
        format="%.2f",
        key=key,
    )


def _summary_table(rows: list[tuple[str, str]]) -> None:
    st.dataframe(
        pd.DataFrame(rows, columns=["Indicador", "Valor"]),
        width="stretch",
        hide_index=True,
        column_config={
            "Indicador": st.column_config.TextColumn("Indicador", width="medium"),
            "Valor": st.column_config.TextColumn("Valor", width="large"),
        },
    )


def _show_theoretical_summary(result: TheoreticalResult) -> None:
    mean_of_means = float(result.means.mean())
    empirical_se = float(result.means.std(ddof=1))
    mean_variance = float(result.variances.mean())
    theoretical_se = float(np.sqrt(result.true_variance / result.sample_size))
    rows = [
        ("Media verdadera", format_number(result.true_mean)),
        ("Media de las medias", format_number(mean_of_means)),
        ("Error estándar teórico", format_number(theoretical_se)),
        ("Error estándar empírico", format_number(empirical_se)),
        ("Cobertura de los IC de la media", f"{100 * (1 - result.mean_misses.mean()):.1f}%"),
        ("Varianza verdadera", format_number(result.true_variance)),
        ("Media de las varianzas S²", format_number(mean_variance)),
        ("Cobertura de los IC de la varianza", f"{100 * (1 - result.variance_misses.mean()):.1f}%"),
    ]
    _summary_table(rows)
    if result.warning:
        st.warning(result.warning)


def _show_manual_interval_summary(result: ManualIntervalResult) -> None:
    rows = [
        ("Número de observaciones", str(result.n)),
        ("Media muestral", format_number(result.sample_mean)),
        ("Varianza muestral S²", format_number(result.sample_variance)),
        ("Método", result.method),
    ]
    if result.standard_error is not None:
        rows.append(("Error estándar", format_number(result.standard_error)))
    else:
        rows.append(("Grados de libertad", str(result.degrees_freedom)))
    rows.append(
        (
            f"IC {result.confidence * 100:.1f}%",
            f"[{format_number(result.low)}, {format_number(result.high)}]",
        )
    )
    _summary_table(rows)
    if result.warning:
        st.warning(result.warning)


def _show_manual_simulation_summary(result: ManualSimulationResult) -> None:
    rows = [
        ("Parámetro de referencia", format_number(result.reference)),
        ("Media de las estimaciones", format_number(float(result.estimates.mean()))),
        ("Sesgo observado", format_number(float(result.estimates.mean() - result.reference))),
        ("Cobertura observada", f"{100 * (1 - result.misses.mean()):.1f}%"),
        ("Método de simulación", result.simulation_method),
        ("Método de intervalo", result.interval_name),
    ]
    _summary_table(rows)
    if result.warning:
        st.warning(result.warning)


def _plot(fig, key: str) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        height=520,
        theme="streamlit",
        config=PLOTLY_CONFIG,
        key=key,
    )


def theoretical_mode() -> None:
    st.subheader("Muestreo desde una población teórica")
    st.caption(
        "Seleccione la distribución poblacional, genere muestras repetidas y estudie las distribuciones muestrales de la media y de la varianza."
    )

    with st.expander("Configuración de la simulación", expanded=True):
        distribution_group = st.segmented_control(
            "Tipo de distribución",
            options=tuple(DISTRIBUTION_GROUPS),
            default="Continuas",
            selection_mode="single",
            width="stretch",
            key="distribution_group",
        ) or "Continuas"

        distribution_options = DISTRIBUTION_GROUPS[distribution_group]
        distribution_default = "Normal" if distribution_group == "Continuas" else "Bernoulli"
        distribution_name = st.pills(
            "Distribución de origen",
            options=distribution_options,
            default=distribution_default,
            selection_mode="single",
            width="stretch",
            key=f"distribution_choice_{distribution_group}",
            help="Selección mediante botones: no abre el teclado virtual en teléfonos.",
        ) or distribution_default

        parameter_specs = DISTRIBUTION_SPECS[distribution_name]
        parameter_columns = st.columns(min(len(parameter_specs), 3))
        parameters = {}
        for index, spec in enumerate(parameter_specs):
            with parameter_columns[index % len(parameter_columns)]:
                parameters[spec.key] = _number_input_for_parameter(distribution_name, spec)

        size_column, repetitions_column = st.columns(2)
        with size_column:
            sample_size = st.number_input(
                "Tamaño de cada muestra",
                min_value=2,
                max_value=MAX_SAMPLE_SIZE,
                value=30,
                step=1,
                format="%d",
            )
        with repetitions_column:
            repetitions = st.number_input(
                "Número de repeticiones",
                min_value=10,
                max_value=MAX_REPETITIONS,
                value=200,
                step=10,
                format="%d",
            )

        confidence_column, seed_column = st.columns(2)
        with confidence_column:
            confidence = st.number_input(
                "Nivel de confianza (%)",
                min_value=50.0,
                max_value=99.9,
                value=95.0,
                step=0.5,
                format="%.1f",
            )
        with seed_column:
            seed = st.number_input(
                "Semilla aleatoria",
                min_value=0,
                value=2026,
                step=1,
                format="%d",
            )

        mean_method_label = st.pills(
            "Intervalo para la media",
            options=(
                "Z: usar la varianza poblacional",
                "t: estimar la varianza con la muestra",
            ),
            default="Z: usar la varianza poblacional",
            selection_mode="single",
            width="stretch",
            help="Estos botones no contienen un campo de búsqueda.",
        ) or "Z: usar la varianza poblacional"

        st.caption(
            f"Límite preventivo: hasta {MAX_TOTAL_DRAWS:,} observaciones simuladas por ejecución."
        )
        run = st.button("Simular muestreo", type="primary", width="stretch")

    if run:
        try:
            with st.spinner("Generando muestras y calculando intervalos..."):
                st.session_state.theoretical_result = simulate_theoretical(
                    distribution_name=distribution_name,
                    parameters=parameters,
                    sample_size=int(sample_size),
                    repetitions=int(repetitions),
                    confidence_percent=float(confidence),
                    seed=int(seed),
                    mean_method="z" if mean_method_label.startswith("Z") else "t",
                )
        except ValueError as error:
            st.error(str(error))
            st.session_state.theoretical_result = None

    result: TheoreticalResult | None = st.session_state.theoretical_result
    if result is None:
        st.info("Configure el experimento y pulse **Simular muestreo**.")
        return

    st.markdown("### Resumen numérico")
    _show_theoretical_summary(result)

    st.markdown("### 1. Intervalos de confianza de las medias")
    _plot(
        interval_figure(
            result.means,
            result.mean_lows,
            result.mean_highs,
            result.mean_misses,
            result.true_mean,
            f"Medias muestrales e intervalos — {result.mean_interval_name}",
            "Media muestral",
        ),
        "theory_mean_intervals",
    )

    st.markdown("### 2. Histograma de las medias muestrales")
    _plot(mean_histogram_figure(result), "theory_mean_histogram")

    st.markdown("### 3. Intervalos de confianza de las varianzas")
    _plot(
        interval_figure(
            result.variances,
            result.variance_lows,
            result.variance_highs,
            result.variance_misses,
            result.true_variance,
            "Varianzas muestrales e intervalos χ²",
            "Varianza muestral S²",
        ),
        "theory_variance_intervals",
    )

    st.markdown("### 4. Histograma de las varianzas muestrales")
    _plot(variance_histogram_figure(result), "theory_variance_histogram")


def manual_mode() -> None:
    st.subheader("Datos manuales e intervalos de confianza")
    st.caption(
        "Ingrese una observación por fila. La planilla permite agregar o eliminar filas y pegar una columna copiada desde una hoja de cálculo."
    )

    example_column, clear_column = st.columns(2)
    with example_column:
        load_example = st.button("Cargar datos de ejemplo", width="stretch")
    with clear_column:
        clear_table = st.button("Limpiar planilla", width="stretch")

    if load_example:
        _replace_manual_data([12.1, 10.8, 11.6, 13.0, 12.4, 11.2, 12.7, 10.9, 11.8, 12.5])
        st.rerun()
    if clear_table:
        _replace_manual_data()
        st.rerun()

    edited = st.data_editor(
        st.session_state.manual_frame,
        key=f"manual_editor_{st.session_state.editor_version}",
        width="stretch",
        height="auto",
        hide_index=False,
        num_rows="dynamic",
        placeholder="Ingrese un valor",
        column_config={
            "Observación": st.column_config.NumberColumn(
                "Observación",
                help="Puede usar punto decimal y pegar una columna desde Excel o Google Sheets.",
                format="%.4f",
            )
        },
        on_change=_clear_manual_results,
    )
    st.session_state.manual_frame = edited
    values = pd.to_numeric(edited["Observación"], errors="coerce").dropna().to_numpy(dtype=float)
    st.caption(f"Observaciones válidas ingresadas: {values.size}")

    with st.expander("Configuración del intervalo y de la simulación", expanded=True):
        target = st.segmented_control(
            "Parámetro de interés",
            options=("Media", "Varianza"),
            default="Media",
            selection_mode="single",
            width="stretch",
        ) or "Media"

        confidence_column, variance_status_column = st.columns(2)
        with confidence_column:
            confidence = st.number_input(
                "Nivel de confianza (%)",
                min_value=50.0,
                max_value=99.9,
                value=95.0,
                step=0.5,
                format="%.1f",
                key="manual_confidence",
            )
        with variance_status_column:
            known_variance = st.checkbox("La varianza poblacional es conocida")

        variance_value = None
        if known_variance:
            variance_value = st.number_input(
                "Varianza poblacional σ²",
                min_value=0.0001,
                value=1.00,
                step=0.10,
                format="%.4f",
            )

        st.markdown("#### Simulación basada en la muestra")
        simulation_method = st.pills(
            "Modelo generador",
            options=("Normal ajustada", "Bootstrap"),
            default="Normal ajustada",
            selection_mode="single",
            width="stretch",
            help=(
                "Normal ajustada usa la media y la varianza de referencia. "
                "Bootstrap remuestrea directamente las observaciones ingresadas."
            ),
        ) or "Normal ajustada"

        default_n = max(2, int(values.size))
        sample_size_column, repetitions_column = st.columns(2)
        with sample_size_column:
            sample_size = st.number_input(
                "Tamaño de las muestras simuladas",
                min_value=2,
                max_value=MAX_SAMPLE_SIZE,
                value=default_n,
                step=1,
                format="%d",
                key="manual_sample_size",
            )
        with repetitions_column:
            repetitions = st.number_input(
                "Número de repeticiones",
                min_value=10,
                max_value=MAX_REPETITIONS,
                value=200,
                step=10,
                format="%d",
                key="manual_repetitions",
            )

        seed = st.number_input(
            "Semilla aleatoria",
            min_value=0,
            value=2026,
            step=1,
            format="%d",
            key="manual_seed",
        )

        calculate_column, simulate_column = st.columns(2)
        with calculate_column:
            calculate = st.button("Calcular intervalo de la muestra", width="stretch")
        with simulate_column:
            simulate = st.button(
                "Simular procesos de muestreo",
                type="primary",
                width="stretch",
            )

    if calculate:
        try:
            st.session_state.manual_interval_result = calculate_manual_interval(
                values=values,
                target=target,
                confidence_percent=float(confidence),
                known_population_variance=variance_value,
            )
        except ValueError as error:
            st.error(str(error))
            st.session_state.manual_interval_result = None

    if simulate:
        try:
            with st.spinner("Simulando muestras e intervalos..."):
                st.session_state.manual_simulation_result = simulate_from_manual_data(
                    values=values,
                    target=target,
                    confidence_percent=float(confidence),
                    known_population_variance=variance_value,
                    simulation_method=simulation_method,
                    sample_size=int(sample_size),
                    repetitions=int(repetitions),
                    seed=int(seed),
                )
        except ValueError as error:
            st.error(str(error))
            st.session_state.manual_simulation_result = None

    interval_result: ManualIntervalResult | None = st.session_state.manual_interval_result
    if interval_result is not None:
        st.markdown("### Intervalo calculado con los datos ingresados")
        _show_manual_interval_summary(interval_result)

    simulation_result: ManualSimulationResult | None = st.session_state.manual_simulation_result
    if simulation_result is None:
        st.info("Ingrese los datos y pulse uno de los botones de cálculo.")
        return

    st.markdown("### Resumen de la simulación")
    _show_manual_simulation_summary(simulation_result)

    y_title = "Media muestral" if simulation_result.target == "Media" else "Varianza muestral S²"
    st.markdown("### 1. Intervalos generados en cada repetición")
    _plot(
        interval_figure(
            simulation_result.estimates,
            simulation_result.lows,
            simulation_result.highs,
            simulation_result.misses,
            simulation_result.reference,
            f"{simulation_result.interval_name}: intervalos por repetición",
            y_title,
        ),
        "manual_intervals",
    )

    st.markdown("### 2. Distribución del estadístico estandarizado")
    _plot(pivot_histogram_figure(simulation_result), "manual_pivot_histogram")


_init_state()

st.title("Simulador interactivo de muestreo e intervalos de confianza")
st.write(
    "Explore el comportamiento de la media y la varianza muestral desde una población teórica, "
    "o calcule y simule intervalos a partir de datos ingresados manualmente."
)

mode = st.segmented_control(
    "Modo de trabajo",
    options=("Población teórica", "Datos manuales e IC"),
    default="Población teórica",
    selection_mode="single",
    width="stretch",
    help="Selección mediante botones, sin campo de búsqueda ni teclado virtual.",
) or "Población teórica"

if mode == "Población teórica":
    theoretical_mode()
else:
    manual_mode()

with st.expander("Supuestos estadísticos y lectura de los gráficos"):
    st.markdown(
        """
        - La varianza muestral se calcula con divisor **n−1**.
        - El intervalo **Z** para la media se utiliza cuando la varianza poblacional es conocida.
        - El intervalo **t** utiliza la varianza estimada mediante la muestra.
        - El intervalo **χ²** para la varianza es exacto cuando la población es normal.
        - En los gráficos de intervalos, el color rojo identifica las repeticiones cuyo intervalo no contiene el valor de referencia.
        - En el modo manual, los histogramas muestran el estadístico estandarizado, porque las distribuciones Z, t y χ² describen ese estadístico y no directamente los estimadores sin estandarizar.
        """
    )
