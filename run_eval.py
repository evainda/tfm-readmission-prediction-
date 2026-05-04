import sys, os
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.train import load_data, split_data
import joblib

print("Loading data...")
X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print(f"Test: {X_test.shape[0]:,} registros")

model = joblib.load("models/xgboost_tuned.pkl")
print("Model loaded:", type(model).__name__)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                              f1_score, accuracy_score, brier_score_loss,
                              classification_report, roc_curve, precision_recall_curve,
                              confusion_matrix, ConfusionMatrixDisplay, calibration_curve)

print(f"\n=== METRICAS FINALES (test set) ===")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print(f"AP:       {average_precision_score(y_test, y_prob):.4f}")
print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
print(f"F1:       {f1_score(y_test, y_pred):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Brier:    {brier_score_loss(y_test, y_prob):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["No readmision","Readmision"]))

# ROC curve
print("Generating figures...")
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_val = roc_auc_score(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, label=f'XGBoost tuneado (AUC = {auc_val:.3f})')
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set_xlabel('Tasa de falsos positivos'); ax.set_ylabel('Tasa de verdaderos positivos')
ax.set_title('Curva ROC — XGBoost tuneado (test set)')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('results/roc_best_model.png', dpi=150); plt.close()
print("  roc_best_model.png OK")

# PR curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ap_val = average_precision_score(y_test, y_prob)
baseline = y_test.mean()
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, lw=2, label=f'XGBoost tuneado (AP = {ap_val:.3f})')
ax.axhline(y=baseline, color='gray', linestyle='--', lw=1, label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Curva Precision-Recall — XGBoost tuneado (test set)')
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('results/pr_curve_best_model.png', dpi=150); plt.close()
print("  pr_curve_best_model.png OK")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=['No reingreso', 'Reingreso'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Matriz de confusión — XGBoost tuneado (umbral 0.5)')
plt.tight_layout(); plt.savefig('results/confusion_matrix_best_model.png', dpi=150); plt.close()
print("  confusion_matrix_best_model.png OK")

# Calibration curve
import numpy as np
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
bs = brier_score_loss(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(prob_pred, prob_true, 's-', lw=2, label=f'XGBoost tuneado (Brier = {bs:.4f})')
ax.plot([0,1],[0,1],'k--', lw=1, label='Calibración perfecta')
ax.set_xlabel('Probabilidad predicha'); ax.set_ylabel('Fracción de positivos observados')
ax.set_title('Curva de calibración — XGBoost tuneado (test set)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('results/calibration_best_model.png', dpi=150); plt.close()
print("  calibration_best_model.png OK")

# Feature importance
import pandas as pd
fi = model.feature_importances_
fi_df = pd.DataFrame({'feature': X.columns, 'importance': fi}).sort_values('importance', ascending=False).head(20)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
ax.set_xlabel('Importancia (ganancia)'); ax.set_title('Top 20 variables por importancia — XGBoost tuneado')
plt.tight_layout(); plt.savefig('results/feature_importance_best_model.png', dpi=150); plt.close()
print("  feature_importance_best_model.png OK")

# SHAP
print("Computing SHAP (sample 2000)...")
import shap
X_sample = X_test.sample(n=2000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
fig, ax = plt.subplots(figsize=(8, 7))
shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
plt.tight_layout(); plt.savefig('results/shap_summary_best_model.png', dpi=150, bbox_inches='tight'); plt.close()
print("  shap_summary_best_model.png OK")

# DCA
print("Computing DCA...")
thresholds = np.linspace(0.01, 0.99, 99)
net_benefit_model = []
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tp = ((y_pred_t == 1) & (y_test == 1)).sum()
    fp = ((y_pred_t == 1) & (y_test == 0)).sum()
    n = len(y_test)
    nb = tp/n - fp/n * (t/(1-t))
    net_benefit_model.append(nb)
prevalence = y_test.mean()
net_benefit_all = [prevalence - (1-prevalence)*t/(1-t) if t < 1 else 0 for t in thresholds]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, net_benefit_model, lw=2, label='XGBoost tuneado')
ax.plot(thresholds, net_benefit_all, lw=1.5, linestyle='--', label='Tratar a todos')
ax.axhline(0, color='gray', lw=1, linestyle=':', label='No tratar a ninguno')
ax.set_xlim(0, 0.8); ax.set_ylim(-0.05, 0.25)
ax.set_xlabel('Umbral de probabilidad'); ax.set_ylabel('Beneficio neto')
ax.set_title('Decision Curve Analysis — XGBoost tuneado (test set)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('results/dca_best_model.png', dpi=150); plt.close()
print("  dca_best_model.png OK")

print("\nDone. All figures saved to results/")
