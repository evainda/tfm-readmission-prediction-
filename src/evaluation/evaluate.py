import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve


def plot_roc_curve(model, X_test, y_test, model_name="Model", ax=None, save_path=None):
    """
    Curva ROC con área bajo la curva (AUC).

    En contexto clínico es la métrica principal porque resume cómo de bien
    separa el modelo los reingresos de los que no reingresan, sin depender
    de un umbral concreto.
    """

    # get probabilities for positive class
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"Curva ROC — {model_name}")
    ax.legend(loc="lower right")

    if owns_fig:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if owns_fig:
        plt.show()

    return ax


def plot_precision_recall_curve(model, X_test, y_test, model_name="Model", ax=None, save_path=None):
    """
    Curva Precision-Recall.

    Complementa la curva ROC en datasets con clases desbalanceadas,
    ya que muestra el trade-off entre precisión y recall directamente
    sobre la clase positiva (readmisión).
    """

    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, label=f"AP = {ap:.3f}", linewidth=2)
    ax.axhline(y=y_test.mean(), color="k", linestyle="--", label=f"Baseline ({y_test.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curva Precision-Recall — {model_name}")
    ax.legend(loc="upper right")

    if owns_fig:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if owns_fig:
        plt.show()

    return ax


def plot_confusion_matrix(model, X_test, y_test, model_name="Model", ax=None, save_path=None):
    """
    Matriz de confusión con etiquetas legibles.

    Muestra los verdaderos positivos, falsos positivos, verdaderos negativos
    y falsos negativos, lo que permite evaluar el impacto clínico de los
    errores del modelo (especialmente los falsos negativos, que son los
    pacientes en riesgo no detectados).
    """

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No readmisión", "Readmisión"],
        yticklabels=["No readmisión", "Readmisión"]
    )
    ax.set_title(f"Matriz de confusión — {model_name}")
    ax.set_ylabel("Valor real")
    ax.set_xlabel("Predicción")

    if owns_fig:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if owns_fig:
        plt.show()

    return ax


def plot_feature_importance(model, feature_names, model_name="Model", top_n=20, save_path=None):
    """
    Importancia de variables del modelo.

    Para modelos basados en árboles (RF, XGBoost, LightGBM) usa la
    importancia por ganancia de información (gain). Para Regresión Logística
    usa los coeficientes absolutos (escalados por StandardScaler dentro del Pipeline,
    por lo que son comparables en magnitud).

    XGBoost usa importance_type='gain' (configurado en get_models()) en lugar
    del valor por defecto 'weight' (número de splits), que está sesgado hacia
    variables de alta cardinalidad.

    Parámetros
    ----------
    top_n : int — número de variables más importantes a mostrar
    """

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline (Logistic Regression dentro de Pipeline)
        clf = model.named_steps["clf"]
        importances = np.abs(clf.coef_[0])
    else:
        raise TypeError(
            f"El modelo '{model_name}' no expone feature_importances_ ni named_steps. "
            "Usa permutation_importance para este tipo de modelo."
        )

    if len(feature_names) != len(importances):
        raise ValueError(
            f"feature_names tiene {len(feature_names)} elementos pero el modelo "
            f"tiene {len(importances)} importancias. Asegúrate de pasar X.columns."
        )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    sns.barplot(data=importance_df, y="feature", x="importance", ax=ax)
    ax.set_title(f"Top {top_n} variables más importantes — {model_name}")
    ax.set_xlabel("Importancia (gain)")
    ax.set_ylabel("")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()

    return importance_df


def plot_calibration_curve(model, X_test, y_test, model_name="Model", ax=None, save_path=None):
    """
    Curva de calibración (reliability diagram).

    Compara la probabilidad predicha por el modelo con la frecuencia
    real de readmisión observada. Un modelo bien calibrado es importante
    en contexto clínico: permite usar las probabilidades directamente
    como estimaciones de riesgo.

    Un modelo por encima de la diagonal sobreestima el riesgo;
    por debajo, lo subestima.
    """

    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(prob_pred, prob_true, marker="o", label=model_name, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")
    ax.set_xlabel("Probabilidad media predicha")
    ax.set_ylabel("Fracción de positivos reales")
    ax.set_title(f"Curva de calibración — {model_name}")
    ax.legend(loc="upper left")

    if owns_fig:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if owns_fig:
        plt.show()

    return ax


def get_classification_report(model, X_test, y_test):
    """
    Devuelve el classification report como DataFrame.
    """

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=["No readmisión", "Readmisión"],
        output_dict=True
    )

    return pd.DataFrame(report).T.round(3)


def plot_shap_summary(model, X_sample, model_name="Model", max_display=20, save_path=None):
    """
    SHAP beeswarm plot para modelos basados en árboles.

    SHAP explica cuánto contribuye cada variable a cada predicción concreta.
    A diferencia de la importancia por ganancia, tiene en cuenta cómo interactúan
    las variables entre sí. Los valores tienen signo: positivo empuja hacia predecir
    reingreso, negativo hacia no reingreso.

    Se recomienda pasar una muestra representativa del test set (p.ej. 1 000 filas)
    en lugar del conjunto completo para reducir el tiempo de cómputo.

    Parámetros
    ----------
    X_sample   : pd.DataFrame — subconjunto de datos sobre el que calcular SHAP
    max_display: int — número máximo de variables a mostrar (default 20)
    save_path  : str — ruta donde guardar la figura (opcional)

    Devuelve
    --------
    pd.DataFrame con importancia SHAP media absoluta por variable (top max_display)
    """
    import shap
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        raise TypeError(
            "plot_shap_summary solo es compatible con modelos de árbol (RF, XGBoost, LightGBM). "
            "Para Logistic Regression usa shap.LinearExplainer."
        )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Para clasificación binaria los SHAP values pueden tener shape (n, p, 2).
    # Tomamos la clase positiva (índice 1 = readmisión).
    if shap_values.values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.title(f"SHAP Summary Plot — {model_name}", pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    mean_abs_shap = (
        pd.DataFrame({
            "feature":        X_sample.columns,
            "mean_abs_shap":  np.abs(sv.values).mean(axis=0)
        })
        .sort_values("mean_abs_shap", ascending=False)
        .head(max_display)
        .reset_index(drop=True)
    )

    return mean_abs_shap


def brier_score(model, X_test, y_test):
    """
    Calcula el Brier Score del modelo.

    Mide el error cuadrático medio entre las probabilidades predichas y los
    valores reales (0/1). Cuanto más bajo, mejor. El baseline de referencia
    es predecir la prevalencia para todos: p*(1-p).
    """
    from sklearn.metrics import brier_score_loss

    y_prob = model.predict_proba(X_test)[:, 1]
    score = brier_score_loss(y_test, y_prob)
    baseline = y_test.mean() * (1 - y_test.mean())

    print(f"Brier Score:  {score:.4f}")
    print(f"Baseline:     {baseline:.4f}  (prediciendo siempre la prevalencia)")
    print(f"Mejora:       {baseline - score:.4f}")

    return score


def expected_calibration_error(model, X_test, y_test, n_bins=10):
    """
    Expected Calibration Error (ECE).

    Divide las probabilidades predichas en n_bins intervalos y compara
    la confianza media de cada bin con la fracción real de positivos.
    ECE = suma ponderada de esas diferencias absolutas.

    Cuanto más cercano a 0, mejor calibrado está el modelo.
    Es más informativa que el Brier Score para detectar dónde falla
    la calibración (bins de alta o baja probabilidad).
    """
    from sklearn.calibration import calibration_curve

    y_prob = model.predict_proba(X_test)[:, 1]
    y_arr  = np.asarray(y_test)

    # Calculamos ECE manualmente para garantizar alineamiento exacto entre
    # pesos, prob_true y prob_pred (calibration_curve omite bins vacíos y
    # np.histogram siempre devuelve n_bins, por lo que combinarlos puede
    # desalinear los arrays si los bins vacíos no coinciden exactamente).
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids   = np.digitize(y_prob, bin_edges[1:-1])  # 0-indexed bin per sample

    weights, prob_true, prob_pred = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        weights.append(mask.sum() / len(y_prob))
        prob_true.append(y_arr[mask].mean())
        prob_pred.append(y_prob[mask].mean())

    weights    = np.array(weights)
    prob_true  = np.array(prob_true)
    prob_pred  = np.array(prob_pred)

    ece = float(np.sum(weights * np.abs(prob_true - prob_pred)))

    baseline_prob = float(y_arr.mean())
    ece_baseline  = float(np.sum(weights * np.abs(prob_true - baseline_prob)))

    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(f"ECE baseline (prevalencia fija):  {ece_baseline:.4f}")
    print(f"Número de bins:                   {n_bins}")

    return ece


def plot_dca(model, X_test, y_test, model_name="Model", thresholds=None, save_path=None):
    """
    Decision Curve Analysis (DCA).

    Compara el modelo frente a dos estrategias simples: tratar a todos los
    pacientes o no tratar a ninguno. Si la curva del modelo queda por encima
    de las dos, el modelo aporta algo real. Sirve para ver en qué rango de
    umbrales tiene sentido usarlo clínicamente.

    Beneficio neto = TP/n - (t / (1-t)) * FP/n
    """

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.75, 150)

    y_prob = model.predict_proba(X_test)[:, 1]
    n = len(y_test)
    prevalence = y_test.mean()

    nb_model, nb_all = [], []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tp = ((y_pred_t == 1) & (y_test == 1)).sum()
        fp = ((y_pred_t == 1) & (y_test == 0)).sum()
        nb_model.append(tp / n - (t / (1 - t)) * fp / n)
        nb_all.append(prevalence - (t / (1 - t)) * (1 - prevalence))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, nb_model, label=model_name, linewidth=2)
    ax.plot(thresholds, nb_all, label="Tratar a todos", linestyle="--", color="gray")
    ax.axhline(0, color="black", linestyle=":", label="No tratar a nadie")
    ax.set_xlabel("Umbral de probabilidad")
    ax.set_ylabel("Beneficio neto")
    ax.set_title(f"Decision Curve Analysis — {model_name}")
    ax.legend()
    ax.set_xlim(0, 0.75)
    ax.set_ylim(-0.05, prevalence * 1.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    return pd.DataFrame({
        "threshold":        thresholds,
        "net_benefit_model": nb_model,
        "net_benefit_all":   nb_all,
    })


def metrics_by_subgroup(model, X, y, groups, min_samples=50):
    """
    Calcula métricas de equidad por subgrupo: ROC-AUC (discriminación) y
    Brier Score vs. baseline (calibración).

    Parámetros
    ----------
    groups      : pd.Series — variable de agrupación (mismo índice que X e y)
    min_samples : int — grupos con menos registros se excluyen (default 50)
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss

    y_prob = model.predict_proba(X)[:, 1]
    y_arr  = np.asarray(y)
    g_arr  = np.asarray(groups)  # grupos como array para poder indexar con mask

    rows = []
    for g in sorted(set(g_arr), key=str):
        mask = g_arr == g
        if mask.sum() < min_samples:
            continue
        if len(np.unique(y_arr[mask])) < 2:
            continue
        prev = float(y_arr[mask].mean())
        rows.append({
            "Subgrupo":       str(g),
            "N":              int(mask.sum()),
            "Prevalencia":    round(prev, 3),
            "ROC-AUC":        round(roc_auc_score(y_arr[mask], y_prob[mask]), 3),
            "Brier-Score":    round(brier_score_loss(y_arr[mask], y_prob[mask]), 3),
            "Brier-Baseline": round(prev * (1 - prev), 3),
        })

    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def plot_subgroup_auc(df_metrics, group_label, save_path=None):
    """
    Gráfico de barras horizontal de ROC-AUC por subgrupo.

    Los subgrupos con ROC-AUC < 0.60 se resaltan en rojo para facilitar
    la identificación visual de grupos con rendimiento reducido.
    """
    n = len(df_metrics)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.5 + 1)))

    colors = ["#d62728" if v < 0.60 else "#1f77b4" for v in df_metrics["ROC-AUC"]]
    ax.barh(df_metrics["Subgrupo"].astype(str), df_metrics["ROC-AUC"], color=colors)

    mean_auc = df_metrics["ROC-AUC"].mean()
    ax.axvline(x=mean_auc, color="gray", linestyle="--", alpha=0.8,
               label=f"Media = {mean_auc:.3f}")

    ax.set_xlabel("ROC-AUC")
    ax.set_title(f"ROC-AUC por {group_label}")
    ax.set_xlim(0.45, 0.85)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return ax


def plot_subgroup_brier(df_metrics, group_label, save_path=None):
    """
    Gráfico de barras agrupadas: Brier Score del modelo vs. baseline por subgrupo.

    El baseline es p*(1-p) (predecir siempre la prevalencia del subgrupo).
    Si la barra del modelo supera al baseline, el modelo calibra peor que
    no usar ningún modelo para ese grupo.
    """
    df = df_metrics.sort_values("Brier-Score").reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.5 + 1)))
    ax.barh(x - width / 2, df["Brier-Score"],    width, label="Modelo",    color="#1f77b4")
    ax.barh(x + width / 2, df["Brier-Baseline"], width, label="Baseline",  color="#aec7e8")

    ax.set_yticks(x)
    ax.set_yticklabels(df["Subgrupo"].astype(str))
    ax.set_xlabel("Brier Score (menor = mejor)")
    ax.set_title(f"Paridad de calibración (Brier Score) por {group_label}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return ax


def plot_shap_waterfall(model, X_sample, y_prob=None, idx_high=None, idx_low=None,
                        model_name="Model", max_display=12,
                        save_path_high=None, save_path_low=None):
    """
    Waterfall plots SHAP para un paciente de alto riesgo y uno de bajo riesgo.

    Complementa el beeswarm global mostrando cómo contribuye cada variable a
    una predicción individual concreta. Útil en contexto clínico para justificar
    una alerta ante el equipo médico.

    Parámetros
    ----------
    X_sample      : pd.DataFrame — subconjunto de datos (mismo usado en beeswarm)
    y_prob        : array — probabilidades predichas sobre X_sample (opcional;
                    si None se calculan internamente)
    idx_high/low  : int — índices explícitos del paciente alto/bajo riesgo
                    (si None se usan argmax/argmin de y_prob)
    max_display   : int — número máximo de variables a mostrar por gráfico
    save_path_*   : str — rutas donde guardar las figuras (opcional)

    Devuelve
    --------
    dict con keys 'idx_high', 'idx_low', 'prob_high', 'prob_low'
    """
    import shap

    if y_prob is None:
        y_prob = model.predict_proba(X_sample)[:, 1]

    if idx_high is None:
        idx_high = int(y_prob.argmax())
    if idx_low is None:
        idx_low = int(y_prob.argmin())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    if shap_values.values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    for idx, label, save_path in [
        (idx_high, "alto riesgo", save_path_high),
        (idx_low,  "bajo riesgo", save_path_low),
    ]:
        plt.figure()
        shap.plots.waterfall(sv[idx], max_display=max_display, show=False)
        plt.title(
            f"{model_name} — Paciente de {label}  "
            f"(P(reingreso) = {y_prob[idx]:.2f})",
            fontsize=10
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return {
        "idx_high":  idx_high,
        "idx_low":   idx_low,
        "prob_high": round(float(y_prob[idx_high]), 3),
        "prob_low":  round(float(y_prob[idx_low]),  3),
    }


def threshold_analysis(model, X_test, y_test, thresholds=None, optimize_for="F1-score"):
    """
    Analiza el efecto del umbral de decisión sobre precision, recall y F1.

    En contexto clínico, bajar el umbral aumenta el recall (se detectan
    más pacientes en riesgo) a costa de más falsos positivos.
    Permite elegir un umbral según cuántos falsos positivos se está dispuesto a asumir.

    Parámetros
    ----------
    thresholds    : array-like — umbrales a evaluar (default: 0.1 a 0.9 en pasos de 0.05)
    optimize_for  : str — métrica para destacar el umbral óptimo (default: "F1-score")
    """

    from sklearn.metrics import precision_score, recall_score, f1_score

    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    y_prob = model.predict_proba(X_test)[:, 1]
    rows = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            "Threshold": round(t, 2),
            "Precision": precision_score(y_test, y_pred_t, zero_division=0),
            "Recall":    recall_score(y_test, y_pred_t, zero_division=0),
            "F1-score":  f1_score(y_test, y_pred_t, zero_division=0),
        })

    df = pd.DataFrame(rows).set_index("Threshold").round(3)

    # Umbral óptimo según la métrica seleccionada
    optimal_threshold = df[optimize_for].idxmax()

    fig, ax = plt.subplots(figsize=(9, 4))
    df.plot(ax=ax)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Umbral 0.5")
    ax.axvline(x=optimal_threshold, color="red", linestyle="--", alpha=0.7,
               label=f"Umbral óptimo {optimize_for}: {optimal_threshold}")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Métrica")
    ax.set_title("Precision / Recall / F1 según umbral de decisión")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nUmbral óptimo ({optimize_for}): {optimal_threshold}")
    print(df.loc[optimal_threshold].to_string())

    return df
