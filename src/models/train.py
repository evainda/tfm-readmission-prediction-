import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import DATA_PROCESSED, TARGET_VARIABLE, TEST_SIZE, RANDOM_STATE
from src.data.load import load_csv


def load_data():
    """
    Carga el dataset final generado por el pipeline de preprocesamiento.

    Devuelve X (variables predictoras) e y (variable objetivo).
    """

    path = DATA_PROCESSED / "model_dataset.csv"
    df = load_csv(path)

    assert TARGET_VARIABLE in df.columns, (
        f"Variable objetivo '{TARGET_VARIABLE}' no encontrada en el dataset. "
        "Verifica que el preprocesamiento se ha ejecutado correctamente."
    )

    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]

    print(f"Dataset cargado: {X.shape[0]:,} registros, {X.shape[1]} variables. "
          f"Tasa positivos: {y.mean():.3f}")

    return X, y


def split_data(X, y):
    """
    División estratificada en entrenamiento y test.

    Usa TEST_SIZE y RANDOM_STATE definidos en config.py para garantizar
    reproducibilidad y mantener la proporción de clases en ambos subconjuntos.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def get_models(scale_pos_weight=1.0):
    """
    Devuelve el diccionario de modelos a entrenar.

    - Logistic Regression: baseline lineal interpretable.
    - Random Forest: ensemble de árboles, captura relaciones no lineales.
    - XGBoost: gradient boosting, robusto con datos tabulares.
    - LightGBM: gradient boosting optimizado en velocidad y memoria.

    Todos los modelos usan class_weight='balanced' (o scale_pos_weight en XGBoost)
    para compensar el desbalanceo de clases y mejorar el recall sobre la clase
    positiva (readmisión), que es la clase de interés clínico.

    Parámetros
    ----------
    scale_pos_weight : float — ratio negatives/positives para XGBoost.
        Calculado en run_all_models() a partir de y_train.
    """

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            importance_type="gain",
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            class_weight="balanced",
            importance_type="gain",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
    }

    return models


def get_metrics(y_true, y_pred, y_prob):
    """
    Calcula las métricas de evaluación estándar para clasificación binaria.

    En predicción de readmisión clínica el recall y el ROC-AUC son las
    métricas más relevantes: el recall mide la capacidad de detectar
    pacientes en riesgo real, y el ROC-AUC la capacidad discriminativa
    global del modelo independientemente del umbral de decisión.

    Average Precision (PR-AUC) complementa el ROC-AUC en problemas con
    clases desbalanceadas, ya que se centra exclusivamente en la clase
    positiva y no incluye los verdaderos negativos en su cálculo.

    Parámetros
    ----------
    y_true : array-like — etiquetas reales
    y_pred : array-like — predicciones binarizadas (umbral 0.5)
    y_prob : array-like — probabilidades de la clase positiva
    """

    return {
        "Accuracy":      accuracy_score(y_true, y_pred),
        "Precision":     precision_score(y_true, y_pred, zero_division=0),
        "Recall":        recall_score(y_true, y_pred, zero_division=0),
        "F1-score":      f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":       roc_auc_score(y_true, y_prob),
        "Avg Precision": average_precision_score(y_true, y_prob)
    }


def train_evaluate(name, model, X_train, X_test, y_train, y_test):
    """
    Entrena un modelo y devuelve sus métricas sobre el conjunto de test.

    Parámetros
    ----------
    name    : str — nombre del modelo (para el log)
    model   : estimador sklearn-compatible
    X_train, X_test, y_train, y_test : splits de datos

    Devuelve
    --------
    dict con el nombre del modelo, todas las métricas y el tiempo de entrenamiento.
    """

    print(f"  Training {name}...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = get_metrics(y_test, y_pred, y_prob)
    metrics["Model"] = name
    metrics["Train time (s)"] = round(train_time, 1)

    return metrics


def run_all_models(X_train, X_test, y_train, y_test):
    """
    Entrena todos los modelos definidos en get_models() y devuelve
    una tabla comparativa ordenada por ROC-AUC descendente.

    Calcula scale_pos_weight a partir de y_train para pasar a XGBoost,
    garantizando que el ajuste de clase se basa únicamente en datos de
    entrenamiento y no introduce fuga de información del test set.

    Devuelve
    --------
    results_df : pd.DataFrame con métricas de cada modelo
    trained_models : dict con los modelos ya entrenados
    """

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_models(scale_pos_weight=scale_pos_weight)

    results = []
    trained_models = {}

    print("Training models...")
    for name, model in models.items():
        metrics = train_evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        trained_models[name] = model

    results_df = (
        pd.DataFrame(results)
        .set_index("Model")
        .sort_values(["ROC-AUC", "Avg Precision"], ascending=False)
        .round(4)
    )

    return results_df, trained_models


def tune_model(model_name, X_train, y_train, n_iter=20, cv=3):
    """
    Optimización de hiperparámetros mediante RandomizedSearchCV.

    Explora n_iter combinaciones aleatorias del espacio de hiperparámetros
    usando validación cruzada estratificada interna (cv folds) sobre los
    datos de entrenamiento. El scoring es ROC-AUC, consistente con la
    métrica principal del proyecto.

    Se aplica exclusivamente sobre X_train/y_train para evitar data leakage:
    el test set no participa en ningún momento de la búsqueda.

    Parámetros
    ----------
    model_name : str  — nombre del modelo a optimizar ('LightGBM' o 'XGBoost')
    n_iter     : int  — combinaciones aleatorias a explorar (default 20)
    cv         : int  — folds de CV interna (default 3, equilibrio velocidad/robustez)

    Devuelve
    --------
    best_estimator : modelo reentrenado con los mejores hiperparámetros
    best_params    : dict con los hiperparámetros óptimos
    cv_results     : dict con media y std del ROC-AUC de CV
    """

    param_distributions = {
        "LightGBM": {
            "n_estimators":       [100, 200, 400, 600],
            "learning_rate":      [0.01, 0.05, 0.1, 0.2],
            "max_depth":          [4, 6, 8, -1],
            "num_leaves":         [31, 63, 127],
            "min_child_samples":  [20, 50, 100],
            "subsample":          [0.7, 0.8, 1.0],
            "bagging_freq":       [1, 5, 10],   # requerido para que subsample tenga efecto en LightGBM
            "colsample_bytree":   [0.7, 0.8, 1.0],
        },
        "XGBoost": {
            "n_estimators":       [100, 200, 400],
            "learning_rate":      [0.01, 0.05, 0.1, 0.2],
            "max_depth":          [4, 6, 8],
            "subsample":          [0.7, 0.8, 1.0],
            "colsample_bytree":   [0.7, 0.8, 1.0],
        },
    }

    if model_name not in param_distributions:
        raise ValueError(
            f"No hay espacio de búsqueda definido para '{model_name}'. "
            f"Modelos soportados: {list(param_distributions.keys())}"
        )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    base_model = get_models(scale_pos_weight=scale_pos_weight)[model_name]
    # Evitar over-subscription: RandomizedSearchCV ya paraleliza a nivel de fold/combinación.
    # Si el modelo interno también usa n_jobs=-1, se satura la CPU.
    base_model.set_params(n_jobs=1)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions[model_name],
        n_iter=n_iter,
        scoring="roc_auc",
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print(f"Optimizando hiperparámetros de {model_name} ({n_iter} iteraciones, CV={cv})...")
    search.fit(X_train, y_train)

    print(f"\nMejores hiperparámetros: {search.best_params_}")
    print(f"ROC-AUC medio (CV interna): {search.best_score_:.4f}")

    # Restaurar paralelismo completo en el modelo final (se desactivó para evitar
    # over-subscription durante RandomizedSearchCV)
    search.best_estimator_.set_params(n_jobs=-1)

    return search.best_estimator_, search.best_params_, {
        "cv_roc_auc_mean": round(search.best_score_, 4),
        "cv_roc_auc_std":  round(
            search.cv_results_["std_test_score"][search.best_index_], 4
        )
    }


def cross_validate_model(model, X, y, cv=5):
    """
    Validación cruzada estratificada (StratifiedKFold) con ROC-AUC.

    Proporciona una estimación más robusta del rendimiento que una
    única partición train/test, al promediar sobre múltiples splits.

    IMPORTANTE: debe llamarse con X_train e y_train, nunca con el dataset
    completo, para evitar que las observaciones del test set participen
    en la estimación de rendimiento (data leakage).

    n_jobs=1 en cross_val_score evita over-subscription de CPU cuando los
    modelos ya usan paralelismo interno (XGBoost, LightGBM, Random Forest).

    Parámetros
    ----------
    model : estimador sklearn-compatible (ya instanciado)
    X, y  : datos de entrenamiento (sin incluir test set)
    cv    : número de folds (default 5)

    Devuelve
    --------
    dict con media y desviación estándar del ROC-AUC
    """

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=1)

    return {
        "cv_roc_auc_mean": scores.mean().round(4),
        "cv_roc_auc_std":  scores.std().round(4)
    }
