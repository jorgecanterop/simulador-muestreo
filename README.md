# Simulador interactivo de muestreo e intervalos de confianza

Aplicación para estudiar distribuciones muestrales de la media y la varianza, calcular intervalos de confianza con datos manuales y simular procesos de muestreo basados en una muestra.

## Funcionalidades

- Muestreo desde 13 distribuciones teóricas.
- Intervalos Z o t para la media.
- Intervalos χ² para la varianza.
- Planilla dinámica para ingresar, agregar, eliminar o pegar observaciones.
- Simulación normal ajustada o bootstrap a partir de datos manuales.
- Gráficos interactivo.

## Ejecución local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Nota estadística

Los intervalos y la referencia χ² para la varianza son exactos bajo normalidad poblacional. Fuera de normalidad se presentan con finalidad didáctica y comparativa.
