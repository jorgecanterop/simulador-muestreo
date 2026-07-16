# Simulador interactivo de muestreo e intervalos de confianza

Aplicación Streamlit para estudiar distribuciones muestrales de la media y la varianza, calcular intervalos de confianza con datos manuales y simular procesos de muestreo basados en una muestra.

## Funcionalidades

- Muestreo desde 13 distribuciones teóricas.
- Intervalos Z o t para la media.
- Intervalos χ² para la varianza.
- Planilla dinámica para ingresar, agregar, eliminar o pegar observaciones.
- Simulación normal ajustada o bootstrap a partir de datos manuales.
- Gráficos Plotly interactivos, responsivos y apilados verticalmente.
- Tema claro y oscuro configurados por separado.
- Diseño compatible con PC, tablet y teléfono.

## Ejecución local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tema y tamaño de letra

La aplicación no incluye selectores propios de tema ni de tamaño de letra. En el menú de Streamlit, seleccione:

**Settings → Theme → Use system setting**

Así, el tema sigue la preferencia clara u oscura del sistema/navegador. El tamaño base no está fijado en `config.toml`, por lo que se conserva la escala predeterminada de Streamlit y puede ajustarse con el zoom o las preferencias de accesibilidad del navegador.

Los gráficos se muestran con `theme="streamlit"`, de modo que también cambian junto con el tema activo.

## Despliegue en Streamlit Community Cloud

1. Suba esta carpeta a un repositorio de GitHub.
2. En Streamlit Community Cloud, seleccione el repositorio y `app.py` como archivo principal.
3. Use Python 3.11 o 3.12.
4. Para integrarlo en Google Sites, use la URL pública con `?embed=true`.
5. No agregue `embed_options=light_theme` ni `embed_options=dark_theme` si desea conservar la adaptación automática.

## Estructura

```text
simulador_muestreo_streamlit/
├── app.py
├── sampling_core.py
├── charts.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── tests/
    └── test_core.py
```

## Nota estadística

Los intervalos y la referencia χ² para la varianza son exactos bajo normalidad poblacional. Fuera de normalidad se presentan con finalidad didáctica y comparativa.
