# Predicción de Reingresos Hospitalarios a 30 días

**TFM · Máster en Inteligencia Artificial · UTAMED 2025/2026**

---

¿Puede un hospital saber, en el momento del alta, qué pacientes van a volver a urgencias antes de que pasen 30 días? Esta fue la pregunta de partida. La respuesta corta es: sí, con matices importantes, y esos matices son gran parte de lo que hace interesante este trabajo.

El proyecto parte de más de 315.000 ingresos reales del hospital Beth Israel Deaconess de Boston (datos MIMIC-IV), compara cuatro algoritmos de machine learning y evalúa el modelo final desde varios ángulos (no solo la métrica de turno) para entender cuándo funciona, cuándo falla y para quién falla más.

---

## Qué hay en este repositorio

```
notebooks/          # Todo el análisis, en orden
  01_eda.ipynb          → exploración de los datos
  02_preprocessing.ipynb → limpieza, features, partición
  03_models.ipynb        → comparación de modelos + tuning
  04_evaluation.ipynb    → evaluación completa del modelo final
  05_subgroup_analysis.ipynb → análisis por subgrupos (equidad)

src/                # Código reutilizable extraído de los notebooks
  data/preprocessing.py
  models/train.py
  evaluation/evaluate.py
  pipeline/train_pipeline.py

models/             # Modelos entrenados (pkl)
  lightgbm_optimizado.pkl
  lightgbm_calibrated.pkl

results/            # Todas las figuras generadas
data/               # raw / interim / processed (los datos no se suben)
```

---

## Dataset

**MIMIC-IV**: base de datos clínica de acceso controlado del MIT.  
Requiere completar un curso de formación en protección de datos y solicitar acceso a través de [PhysioNet](https://physionet.org/content/mimiciv/). Los datos en sí no están incluidos en este repositorio.

El dataset final tiene **315.982 ingresos**, 54 variables y una tasa de reingreso del 34,2%.

---

## Resumen del enfoque

### División de datos

Se usó `GroupShuffleSplit` agrupando por paciente (`subject_id`) para garantizar que ningún paciente aparece a la vez en entrenamiento y en test. Esto es más importante de lo que parece: con divisiones aleatorias estándar, el modelo aprende patrones del historial de un paciente que en producción no existirían. La partición final fue 60/20/20 (train/val/test).

### Modelos comparados

| Modelo | AUC (validación) |
|--------|-----------------|
| LightGBM | 0.651 |
| XGBoost | 0.650 |
| Regresión Logística | 0.633 |
| Random Forest | 0.609 |

LightGBM ganó por poco en discriminación, pero la diferencia más relevante estaba en sensibilidad: Random Forest, aunque parecía competitivo en AUC, habría dejado escapar cuatro de cada cinco reingresos reales.

### Más allá del AUC

- **Calibración**: el modelo con pesos balanceados sobreestima el riesgo. Se aplicó postcalibración isotónica sobre validación, lo que corrige bien el problema.
- **SHAP**: las variables más influyentes son el número de ingresos previos, la duración del ingreso actual y la edad. Tiene sentido clínico y eso da cierta tranquilidad.
- **DCA (Decision Curve Analysis)**: el modelo tiene utilidad clínica neta en el rango de umbrales más habitual para intervenciones preventivas.
- **Análisis por subgrupos**: el rendimiento cae en mayores de 80 años (AUC 0.613) y en algunos grupos diagnósticos pequeños. Es importante saberlo antes de desplegarlo.

---

## Cómo ejecutarlo

### Requisitos

```bash
pip install -r requirements.txt
```

Las versiones principales: `pandas 3.0`, `scikit-learn 1.8`, `lightgbm 4.6`, `xgboost 3.2`, `shap 0.46`.

### Orden de ejecución

Los notebooks están numerados y pensados para ejecutarse en orden. Cada uno guarda lo que necesita el siguiente (datos procesados, modelos, etc.).

1. Coloca los archivos CSV de MIMIC-IV en `data/raw/`
2. Ejecuta los notebooks del 01 al 05 en orden

También puedes abrir los notebooks directamente en Colab:

| Notebook | Colab |
|----------|-------|
| 01 · EDA | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evainda/tfm-readmission-prediction-/blob/main/notebooks/01_eda.ipynb) |
| 02 · Preprocesado | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evainda/tfm-readmission-prediction-/blob/main/notebooks/02_preprocessing.ipynb) |
| 03 · Modelos | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evainda/tfm-readmission-prediction-/blob/main/notebooks/03_models.ipynb) |
| 04 · Evaluación | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evainda/tfm-readmission-prediction-/blob/main/notebooks/04_evaluation.ipynb) |
| 05 · Subgrupos | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evainda/tfm-readmission-prediction-/blob/main/notebooks/05_subgroup_analysis.ipynb) |

---

## Limitaciones conocidas

- Solo datos administrativos: sin signos vitales, analíticas ni notas de enfermería. La mayoría de modelos que superan AUC 0.70 usan ese tipo de información.
- Validación interna únicamente (un solo hospital). Antes de usarlo en otro contexto habría que validarlo allí.
- La calibración del modelo base no es buena. El modelo calibrado (`lightgbm_calibrated.pkl`) es el que debería usarse si las probabilidades individuales importan.
- Agrupación y codificación one-hot calculadas sobre el dataset completo (antes de la partición). Con 315k registros la diferencia es mínima, pero en producción debería hacerse solo sobre train.

---

## Referencia

Johnson, A., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67

---

*Eva Inda · UTAMED · 2026*
