"""
Pipeline completo de entrenamiento end-to-end.

Ejecuta en orden todas las etapas del proyecto:
  1. Preprocesamiento (parte 1: carga y limpieza)
  2. Preprocesamiento (parte 2: target + encoding)
  3. División train/test estratificada
  4. Entrenamiento y comparación de los 4 modelos
  5. Optimización de hiperparámetros de LightGBM
  6. Evaluación final del modelo tuneado
  7. Serialización del modelo final

Uso:
    python -m src.pipeline.train_pipeline
    python -m src.pipeline.train_pipeline --skip-preprocessing
"""

import argparse

from src.data.preprocessing import run_preprocessing_part1, run_preprocessing_part2
from src.models.train import (
    load_data,
    split_data,
    run_all_models,
    tune_model,
    cross_validate_model,
    save_model,
    get_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline completo de entrenamiento")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help=(
            "Omite las etapas de preprocesamiento y carga directamente "
            "el dataset procesado existente en data/processed/model_dataset.csv"
        ),
    )
    parser.add_argument(
        "--tune-model",
        default="LightGBM",
        choices=["LightGBM", "XGBoost"],
        help="Modelo a optimizar con RandomizedSearchCV (default: LightGBM)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Número de iteraciones de RandomizedSearchCV (default: 20)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Número de folds para la búsqueda de hiperparámetros (default: 3)",
    )
    return parser.parse_args()


def run_pipeline(skip_preprocessing=False, tune_model_name="LightGBM", n_iter=20, cv=3):
    """
    Ejecuta el pipeline completo.

    Parámetros
    ----------
    skip_preprocessing : bool
        Si True, omite el preprocesamiento y carga el dataset ya procesado.
    tune_model_name : str
        Nombre del modelo a optimizar ('LightGBM' o 'XGBoost').
    n_iter : int
        Iteraciones de RandomizedSearchCV.
    cv : int
        Folds de validación cruzada interna en RandomizedSearchCV.
    """

    # ------------------------------------------------------------------
    # ETAPA 1: Preprocesamiento
    # ------------------------------------------------------------------
    if not skip_preprocessing:
        print("\n" + "=" * 60)
        print("ETAPA 1 — Preprocesamiento")
        print("=" * 60)

        df = run_preprocessing_part1()
        run_preprocessing_part2(df)
    else:
        print("\n[INFO] Preprocesamiento omitido. Cargando dataset procesado...")

    # ------------------------------------------------------------------
    # ETAPA 2: Carga y división de datos
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 2 — Carga y división de datos")
    print("=" * 60)

    X, y, groups = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, groups)
    groups_train = groups.loc[X_train.index] if groups is not None else None

    print(
        f"Train: {len(X_train):,} registros | "
        f"Val: {len(X_val):,} registros | "
        f"Test: {len(X_test):,} registros | "
        f"Tasa positivos (train): {y_train.mean():.3f}"
    )

    # ------------------------------------------------------------------
    # ETAPA 3: Comparación de modelos base (sobre validación)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 3 — Comparación de modelos")
    print("=" * 60)

    results_df, trained_models = run_all_models(X_train, X_val, y_train, y_val)

    print("\nResultados comparativos (ordenados por ROC-AUC):")
    print(results_df.to_string())

    # ------------------------------------------------------------------
    # ETAPA 4: Optimización del mejor modelo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"ETAPA 4 — Optimización de hiperparámetros ({tune_model_name})")
    print("=" * 60)

    best_model, best_params, cv_metrics = tune_model(
        tune_model_name, X_train, y_train, n_iter=n_iter, cv=cv, groups=groups_train
    )

    print(f"\nMejores parámetros encontrados: {best_params}")
    print(
        f"ROC-AUC (CV interna): "
        f"{cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics['cv_roc_auc_std']:.4f}"
    )

    # ------------------------------------------------------------------
    # ETAPA 5: Validación cruzada del modelo tuneado sobre train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 5 — Validación cruzada del modelo tuneado (5 folds)")
    print("=" * 60)

    cv_scores = cross_validate_model(best_model, X_train, y_train, cv=5, groups=groups_train)
    print(
        f"ROC-AUC (CV 5-fold): "
        f"{cv_scores['cv_roc_auc_mean']:.4f} ± {cv_scores['cv_roc_auc_std']:.4f}"
    )

    # ------------------------------------------------------------------
    # ETAPA 6: Evaluación final en el test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 6 — Evaluación final en el test set (holdout)")
    print("=" * 60)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    final_metrics = get_metrics(y_test, y_pred, y_prob)

    print(f"\nMétricas finales del modelo tuneado ({tune_model_name}):")
    for metric, value in final_metrics.items():
        print(f"  {metric:<15}: {value:.4f}")

    # Comparar con el modelo base del mismo tipo
    if tune_model_name in results_df.index:
        base_auc = results_df.loc[tune_model_name, "ROC-AUC"]
        delta = final_metrics["ROC-AUC"] - base_auc
        print(f"\n  Mejora vs. base {tune_model_name}: {delta:+.4f} ROC-AUC")

    # ------------------------------------------------------------------
    # ETAPA 7: Serialización del modelo final
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 7 — Serialización del modelo")
    print("=" * 60)

    model_filename = tune_model_name.lower().replace(" ", "_") + "_tuned"
    save_model(best_model, model_filename)

    print("\n" + "=" * 60)
    print("Pipeline completado correctamente.")
    print("=" * 60)

    return best_model, results_df, final_metrics


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        skip_preprocessing=args.skip_preprocessing,
        tune_model_name=args.tune_model,
        n_iter=args.n_iter,
        cv=args.cv,
    )
