import time

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, cross_val_score, RandomizedSearchCV, GroupShuffleSplit
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

from src.config import DATA_PROCESSED, TARGET_VARIABLE, TEST_SIZE, RANDOM_STATE, PROJECT_ROOT
from src.data.load import load_csv

MODELS_DIR = PROJECT_ROOT / "models"


def save_model(model, name):
    """Serializa el modelo entrenado en models/<name>.pkl."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Modelo guardado en {path}")
    return path


def load_model(name):
    """Carga un modelo serializado desde models/<name>.pkl."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {path}\n"
            "Ejecuta primero 03_models.ipynb para entrenar y guardar el modelo."
        )
    return joblib.load(path)


def load_data():
    """
    Carga el dataset procesado y devuelve X, y y groups.

    groups es la Serie subject_id para usar con GroupShuffleSplit (split por paciente).
    Si el CSV no contiene subject_id (versión antigua), groups es None y el split
    cae en la lógica estratificada aleatoria.
    """

    path = DATA_PROCESSED / "model_dataset.csv"
    df = load_csv(path)

    if TARGET_VARIABLE not in df.columns:
        raise ValueError(
            f"Variable objetivo '{TARGET_VARIABLE}' no encontrada en el dataset. "
            "Verifica que el preprocesamiento se ha ejecutado correctamente."
        )

    groups = df["subject_id"] if "subject_id" in df.columns else None
    X = df.drop(columns=[TARGET_VARIABLE] + (["subject_id"] if "subject_id" in df.columns else []))
    y = df[TARGET_VARIABLE]

    split_mode = "por paciente (GroupShuffleSplit)" if groups is not None else "aleatorio estratificado"
    print(f"Dataset cargado: {X.shape[0]:,} registros, {X.shape[1]} variables. "
          f"Tasa positivos: {y.mean():.3f}. Split: {split_mode}")

    return X, y, groups


def split_data(X, y, groups=None):
    """
    División en tres conjuntos: train (60%), validación (20%) y test (20%).

    Si se proporcionan groups (subject_id), se usa GroupShuffleSplit para garantizar
    que ningún paciente aparece en más de un conjunto (elimina el leakage por paciente).
    Si groups es None, se aplica la división estratificada aleatoria como fallback.

    La comparación de modelos y el tuning se realizan sobre validación.
    El test set se usa una única vez para la evaluación final.
    """

    if groups is not None:
        # Primera división: trainval / test — ningún paciente cruza la frontera
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        trainval_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_trainval = X.iloc[trainval_idx]
        X_test     = X.iloc[test_idx]
        y_trainval = y.iloc[trainval_idx]
        y_test     = y.iloc[test_idx]
        groups_trainval = groups.iloc[trainval_idx]

        # Segunda división: train / val — tampoco cruzan pacientes
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups=groups_trainval))

        X_train = X_trainval.iloc[train_idx]
        X_val   = X_trainval.iloc[val_idx]
        y_train = y_trainval.iloc[train_idx]
        y_val   = y_trainval.iloc[val_idx]

    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        # 0.25 del 80% restante = 20% del total → split final 60/20/20
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_models(scale_pos_weight=1.0):
    """
    Devuelve los cuatro modelos a comparar con sus configuraciones base.

    Logistic Regression y Random Forest usan class_weight='balanced' (sklearn).
    XGBoost y LightGBM usan scale_pos_weight (parámetro nativo) calculado sobre
    y_train como N_neg/N_pos, siguiendo la recomendación de la documentación oficial.
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
            scale_pos_weight=scale_pos_weight,
            importance_type="gain",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
    }

    return models


def get_metrics(y_true, y_pred, y_prob):
    """
    Calcula las métricas principales para clasificación binaria.

    ROC-AUC y Recall son las métricas clave en este contexto clínico:
    el primero resume cómo de bien separa el modelo reingresos de no reingresos,
    y el segundo cuántos pacientes en riesgo real se detectan.
    """

    return {
        "Accuracy":      accuracy_score(y_true, y_pred),
        "Precision":     precision_score(y_true, y_pred, zero_division=0),
        "Recall":        recall_score(y_true, y_pred, zero_division=0),
        "F1-score":      f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":       roc_auc_score(y_true, y_prob),
        "Avg Precision": average_precision_score(y_true, y_prob)
    }


def train_evaluate(name, model, X_train, X_val, y_train, y_val):
    """Entrena un modelo y devuelve sus métricas sobre el conjunto de validación."""

    print(f"  Training {name}...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = get_metrics(y_val, y_pred, y_prob)
    metrics["Model"] = name
    metrics["Train time (s)"] = round(train_time, 1)

    return metrics


def run_all_models(X_train, X_val, y_train, y_val):
    """
    Entrena los cuatro modelos y devuelve una tabla comparativa ordenada por ROC-AUC.

    El scale_pos_weight de XGBoost se calcula a partir de y_train para
    no introducir información del test set en la configuración del modelo.
    """

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_models(scale_pos_weight=scale_pos_weight)

    results = []
    trained_models = {}

    print("Training models...")
    for name, model in models.items():
        metrics = train_evaluate(name, model, X_train, X_val, y_train, y_val)
        results.append(metrics)
        trained_models[name] = model

    results_df = (
        pd.DataFrame(results)
        .set_index("Model")
        .sort_values(["ROC-AUC", "Avg Precision"], ascending=False)
        .round(4)
    )

    return results_df, trained_models


def tune_model(model_name, X_train, y_train, n_iter=20, cv=3, groups=None):
    """
    Optimización de hiperparámetros del modelo seleccionado mediante
    RandomizedSearchCV sobre el conjunto de entrenamiento.

    Se exploran n_iter combinaciones aleatorias con validación cruzada
    interna de cv folds. El scoring es ROC-AUC. El test set no participa
    en ningún momento de la búsqueda.

    Devuelve el modelo reentrenado con los mejores hiperparámetros,
    el diccionario de parámetros y el ROC-AUC medio de CV.
    """

    param_distributions = {
        "LightGBM": {
            "n_estimators":       [100, 200, 400, 600],
            "learning_rate":      [0.01, 0.05, 0.1, 0.2],
            "max_depth":          [4, 6, 8, -1],
            "num_leaves":         [31, 63, 127],
            "min_child_samples":  [20, 50, 100],
            "subsample":          [0.7, 0.8, 1.0],
            "bagging_freq":       [1, 5, 10],
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
    # Desactivar paralelismo interno para evitar saturación de CPU
    # durante la búsqueda (RandomizedSearchCV ya paraleliza por fold)
    base_model.set_params(n_jobs=1)

    if groups is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=cv)
    else:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions[model_name],
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_splitter,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print(f"Optimizando hiperparámetros de {model_name} ({n_iter} iteraciones, CV={cv})...")
    search.fit(X_train, y_train, groups=groups)

    print(f"\nMejores hiperparámetros: {search.best_params_}")
    print(f"ROC-AUC medio (CV interna): {search.best_score_:.4f}")

    search.best_estimator_.set_params(n_jobs=-1)

    return search.best_estimator_, search.best_params_, {
        "cv_roc_auc_mean": round(search.best_score_, 4),
        "cv_roc_auc_std":  round(
            search.cv_results_["std_test_score"][search.best_index_], 4
        )
    }


def cross_validate_model(model, X, y, cv=5, groups=None):
    """
    Validación cruzada estratificada (5 folds) sobre los datos de entrenamiento.

    Proporciona una estimación más robusta del rendimiento que una
    única partición. Debe llamarse siempre con X_train/y_train,
    nunca con el dataset completo.

    Si se proporcionan groups (subject_id), usa StratifiedGroupKFold para
    evitar que el mismo paciente aparezca en train y test del fold.
    """

    if groups is not None:
        cv_splitter = StratifiedGroupKFold(n_splits=cv)
        scores = cross_val_score(
            model, X, y, cv=cv_splitter, scoring="roc_auc",
            n_jobs=1, groups=groups
        )
    else:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=1)

    return {
        "cv_roc_auc_mean": scores.mean().round(4),
        "cv_roc_auc_std":  scores.std().round(4)
    }
