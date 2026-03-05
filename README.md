# Hospital Readmission Prediction using Machine Learning
# Predicción de Reingresos Hospitalarios mediante Machine Learning
---

# 🇪🇸 Versión en Español

## Descripción

Este repositorio contiene el código y los experimentos desarrollados para el TFM centrado en la predicción de reingresos hospitalarios mediante técnicas de aprendizaje automático aplicadas a registros electrónicos de salud (Electronic Health Records, EHR).

El objetivo del proyecto es desarrollar modelos predictivos capaces de estimar la probabilidad de que un paciente vuelva a ser ingresado en el hospital tras recibir el alta.
---

## Dataset

El proyecto utilizará datos clínicos procedentes de registros electrónicos de salud. El dataset principal considerado para este trabajo es:

**MIMIC-IV Clinical Database**

MIMIC-IV es una base de datos clínica pública que contiene datos sanitarios anonimizados de pacientes ingresados en el hospital Beth Israel Deaconess Medical Center. Incluye información detallada como:

- datos demográficos de los pacientes  
- ingresos hospitalarios  
- diagnósticos médicos  
- procedimientos clínicos  
- resultados de laboratorio  
- medicación prescrita  

El acceso al dataset se realiza a través de PhysioNet y requiere autorización previa.

---

## Estructura del Proyecto

tfm-hospital-readmission/

data/ → datasets originales y procesados
notebooks/ → análisis exploratorio y experimentos
src/ → código reutilizable (preprocesamiento, modelos, evaluación)
results/ → figuras y tablas generadas
models/ → modelos entrenados
docs/ → redacción del trabajo de fin de máster


---

## Metodología

El proyecto seguirá el siguiente pipeline de ML:

1. Análisis exploratorio de datos (EDA)
2. Preprocesamiento de datos
3. Ingeniería de características
4. Entrenamiento de modelos
5. Evaluación de modelos
6. Interpretabilidad de modelos

---

## Modelos

Se evaluarán distintos modelos de aprendizaje automático, entre ellos:

- Regresión Logística
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)

---

## Métricas de Evaluación

El rendimiento de los modelos se evaluará mediante diferentes métricas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

También se analizan aspectos adicionales como la calibración del modelo y la interpretabilidad de las predicciones.

---

## Herramientas

El proyecto se desarrolló utilizando:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- matplotlib
- seaborn
- Jupyter Notebook

---

# 🇬🇧 English Version

## Overview

This repository contains the code and experiments developed for a Master's Thesis focused on predicting hospital readmissions using machine learning techniques applied to Electronic Health Records (EHR).

The goal of the project is to develop predictive models capable of estimating the probability that a patient will be readmitted to the hospital after discharge.

---

## Dataset

The project will use clinical datasets derived from Electronic Health Records (EHR). The primary dataset considered for this research is:

**MIMIC-IV Clinical Database**

MIMIC-IV is a large, publicly available database containing de-identified health data associated with patients admitted to the Beth Israel Deaconess Medical Center. It includes detailed information such as patient demographics, hospital admissions, diagnoses, procedures, laboratory measurements, and prescriptions.

The dataset is distributed through PhysioNet and requires credentialed access.

---

## Project Structure

tfm-hospital-readmission/

data/ → raw and processed datasets
notebooks/ → exploratory analysis and experiments
src/ → reusable code (preprocessing, models, evaluation)
results/ → generated figures and tables
models/ → trained models
docs/ → thesis writing

---

## Methodology

The project follows a typical machine learning pipeline:

1. Exploratory Data Analysis (EDA)
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation
6. Model interpretability

---

## Models

The following models will be evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)

---

## Evaluation Metrics

Model performance will be evaluated using several metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional analyses such as calibration and interpretability techniques may also be included.

---

## Tools

The project will be implemented using:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- matplotlib
- seaborn
- Jupyter Notebook
- Visual Studio Code

 
