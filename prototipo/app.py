"""
Panel Clínico de Riesgo de Reingreso a 30 días
Modelo: LightGBM + postcalibración isotónica (MIMIC-IV)

Ejecución:
    cd tfm-readmission-prediction-
    streamlit run prototipo/app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import shap

# ── reproduced from notebook 04_evaluation.ipynb ──────────────────────────────
class _CalibratedWrapper:
    def __init__(self, base_model, ir):
        self.base_model = base_model
        self.ir = ir

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.ir.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

sys.modules["__main__"]._CalibratedWrapper = _CalibratedWrapper

# ── paths ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATED_PATH = os.path.join(HERE, "..", "models", "lightgbm_calibrated.pkl")
OPTIMIZADO_PATH  = os.path.join(HERE, "..", "models", "lightgbm_optimizado.pkl")

# ── feature columns — mismo orden que en entrenamiento ────────────────────────
FEATURE_COLS = [
    "length_of_stay", "n_diagnoses", "previous_admissions", "age_at_admission", "gender_M",
    "race_ASIAN - CHINESE", "race_BLACK/AFRICAN AMERICAN", "race_BLACK/CAPE VERDEAN",
    "race_HISPANIC OR LATINO", "race_HISPANIC/LATINO - DOMINICAN",
    "race_HISPANIC/LATINO - PUERTO RICAN", "race_OTHER", "race_Other", "race_UNKNOWN",
    "race_WHITE", "race_WHITE - OTHER EUROPEAN", "race_WHITE - RUSSIAN",
    "insurance_Medicare", "insurance_No charge", "insurance_Other",
    "insurance_Private", "insurance_Unknown",
    "marital_status_MARRIED", "marital_status_SINGLE",
    "marital_status_Unknown", "marital_status_WIDOWED",
    "language_English", "language_Other", "language_Russian", "language_Spanish",
    "admission_type_DIRECT EMER.", "admission_type_DIRECT OBSERVATION",
    "admission_type_ELECTIVE", "admission_type_EU OBSERVATION", "admission_type_EW EMER.",
    "admission_type_OBSERVATION ADMIT", "admission_type_SURGICAL SAME DAY ADMISSION",
    "admission_type_URGENT",
    "admission_location_CLINIC REFERRAL", "admission_location_EMERGENCY ROOM",
    "admission_location_INFORMATION NOT AVAILABLE",
    "admission_location_INTERNAL TRANSFER TO OR FROM PSYCH", "admission_location_PACU",
    "admission_location_PHYSICIAN REFERRAL", "admission_location_PROCEDURE SITE",
    "admission_location_TRANSFER FROM HOSPITAL",
    "admission_location_TRANSFER FROM SKILLED NURSING FACILITY",
    "admission_location_WALK-IN/SELF REFERRAL",
    "discharge_location_HOME", "discharge_location_HOME HEALTH CARE",
    "discharge_location_Other", "discharge_location_REHAB",
    "discharge_location_SKILLED NURSING FACILITY", "discharge_location_Unknown",
]

# ── etiquetas clínicas legibles para el médico ────────────────────────────────
CLINICAL_LABELS = {
    "previous_admissions":                         "Ingresos previos (total histórico)",
    "length_of_stay":                              "Días de estancia",
    "age_at_admission":                            "Edad al ingreso",
    "n_diagnoses":                                 "Número de diagnósticos",
    "gender_M":                                    "Sexo masculino",
    "discharge_location_HOME":                     "Alta a domicilio",
    "discharge_location_HOME HEALTH CARE":         "Alta con atención domiciliaria",
    "discharge_location_SKILLED NURSING FACILITY": "Alta a centro residencial/enfermería",
    "discharge_location_REHAB":                    "Alta a rehabilitación",
    "discharge_location_Other":                    "Destino al alta: otro",
    "discharge_location_Unknown":                  "Destino al alta: desconocido",
    "insurance_Medicare":                          "Seguro: Medicare",
    "insurance_No charge":                         "Sin cobertura / sin cargo",
    "insurance_Other":                             "Otro tipo de seguro",
    "insurance_Private":                           "Seguro privado",
    "insurance_Unknown":                           "Cobertura desconocida",
    "admission_type_EW EMER.":                     "Admisión por urgencias",
    "admission_type_DIRECT EMER.":                 "Admisión de emergencia directa",
    "admission_type_ELECTIVE":                     "Ingreso programado (electivo)",
    "admission_type_URGENT":                       "Ingreso urgente",
    "admission_type_SURGICAL SAME DAY ADMISSION":  "Cirugía ambulatoria (mismo día)",
    "admission_type_DIRECT OBSERVATION":           "Admisión en observación directa",
    "admission_type_EU OBSERVATION":               "Observación de urgencias",
    "admission_type_OBSERVATION ADMIT":            "Ingreso en observación",
    "marital_status_SINGLE":                       "Estado civil: soltero/a",
    "marital_status_MARRIED":                      "Estado civil: casado/a",
    "marital_status_WIDOWED":                      "Estado civil: viudo/a",
    "marital_status_Unknown":                      "Estado civil: desconocido",
    "language_English":                            "Idioma: inglés",
    "language_Spanish":                            "Idioma: español",
    "language_Other":                              "Idioma: otro",
    "language_Russian":                            "Idioma: ruso",
    "admission_location_EMERGENCY ROOM":           "Procedencia: urgencias",
    "admission_location_PHYSICIAN REFERRAL":       "Procedencia: derivación médica",
    "admission_location_CLINIC REFERRAL":          "Procedencia: consulta externa",
    "admission_location_TRANSFER FROM HOSPITAL":   "Procedencia: traslado hospitalario",
    "admission_location_TRANSFER FROM SKILLED NURSING FACILITY": "Procedencia: residencia/enfermería",
    "admission_location_WALK-IN/SELF REFERRAL":    "Procedencia: demanda espontánea",
}

# ── recomendaciones clínicas por nivel de riesgo ──────────────────────────────
RECOMMENDATIONS = {
    "BAJO": {
        "border": "#28a745",
        "bg":     "#f0fff4",
        "titulo": "Alta rutinaria recomendada",
        "acciones": [
            "Seguimiento en atención primaria en las próximas 1–2 semanas",
            "Confirmar que el paciente comprende la medicación al alta",
            "Informar por escrito de síntomas de alarma que requieren consulta",
        ],
    },
    "MODERADO": {
        "border": "#f0ad4e",
        "bg":     "#fffdf0",
        "titulo": "Seguimiento reforzado recomendado",
        "acciones": [
            "Llamada de seguimiento telefónico a las 48–72 horas del alta",
            "Verificar adherencia al tratamiento en los primeros 7 días",
            "Valorar necesidad de cuidados domiciliarios o apoyo social",
            "Coordinar con médico de atención primaria antes de firmar el alta",
        ],
    },
    "ALTO": {
        "border": "#dc3545",
        "bg":     "#fff5f5",
        "titulo": "Intervención activa antes del alta",
        "acciones": [
            "Activar protocolo de prevención de reingresos del centro",
            "Involucrar a enfermería de enlace o trabajo social antes del alta",
            "Valorar ingreso en unidad de media estancia o rehabilitación",
            "Programar consulta presencial en las primeras 48–72 horas",
            "Revisar optimización del tratamiento y red de apoyo del paciente",
        ],
    },
}


# ── carga del modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load(CALIBRATED_PATH), True
    except (FileNotFoundError, OSError):
        return joblib.load(OPTIMIZADO_PATH), False


# ── construcción del vector de características ────────────────────────────────
def build_features(age, los, prev_adm, n_diag, gender,
                   race, insurance, marital, language,
                   admission_type, admission_loc, discharge_loc):
    row = {col: 0 for col in FEATURE_COLS}
    row["age_at_admission"]   = age
    row["length_of_stay"]     = los
    row["previous_admissions"] = prev_adm
    row["n_diagnoses"]        = n_diag
    row["gender_M"]           = 1 if gender == "Masculino" else 0

    for prefix, value in [
        ("race_",               race),
        ("marital_status_",     marital),
        ("language_",           language),
        ("admission_type_",     admission_type),
        ("admission_location_", admission_loc),
        ("discharge_location_", discharge_loc),
    ]:
        col = f"{prefix}{value}"
        if col in row:
            row[col] = 1

    if insurance != "Medicaid":          # Medicaid = categoría de referencia
        col = f"insurance_{insurance}"
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[FEATURE_COLS]


# ── gráfico: medidor semicircular ─────────────────────────────────────────────
def plot_gauge(prob):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    # zonas de color: BAJO 0–25%, MODERADO 25–40%, ALTO 40–100%
    # eje angular: π (izquierda, prob=0) → 0 (derecha, prob=1)
    zones = [
        (0.00, 0.25, "#d4edda", "#28a745"),
        (0.25, 0.40, "#fff3cd", "#f0ad4e"),
        (0.40, 1.00, "#f8d7da", "#dc3545"),
    ]
    R_OUT, R_IN = 1.0, 0.60
    for p0, p1, fill, _ in zones:
        a0 = np.pi * (1 - p0)
        a1 = np.pi * (1 - p1)
        th = np.linspace(a0, a1, 120)
        xo, yo = np.cos(th) * R_OUT, np.sin(th) * R_OUT
        xi, yi = np.cos(th[::-1]) * R_IN, np.sin(th[::-1]) * R_IN
        ax.fill(np.concatenate([xo, xi]), np.concatenate([yo, yi]),
                color=fill, zorder=1)

    # arco exterior
    th_all = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(th_all) * R_OUT, np.sin(th_all) * R_OUT,
            color="#aaaaaa", lw=1.2, zorder=2)

    # aguja
    needle_angle = np.pi * (1 - prob)
    nx = np.cos(needle_angle)
    ny = np.sin(needle_angle)
    ax.annotate("",
                xy=(0.82 * nx, 0.82 * ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#222222",
                                lw=2.2, mutation_scale=14),
                zorder=4)

    # centro
    ax.add_patch(plt.Circle((0, 0), 0.065, color="#222222", zorder=5))

    # etiquetas de zona
    for label, x, color in [
        ("BAJO",     -0.88, "#28a745"),
        ("MODERADO",  0.00, "#856404"),
        ("ALTO",      0.88, "#dc3545"),
    ]:
        ax.text(x, -0.16, label, ha="center", va="top",
                fontsize=7.5, fontweight="bold", color=color, zorder=6)

    # porcentaje central
    if prob < 0.25:
        pcolor = "#28a745"
    elif prob < 0.40:
        pcolor = "#856404"
    else:
        pcolor = "#dc3545"

    ax.text(0, -0.50, f"{prob:.0%}", ha="center", va="top",
            fontsize=30, fontweight="bold", color=pcolor, zorder=6)
    ax.text(0, -0.80, "probabilidad de\nreingreso a 30 días", ha="center", va="top",
            fontsize=8, color="#555555", zorder=6)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.05, 1.15)
    plt.tight_layout(pad=0)
    return fig


# ── gráfico: barras de factores ───────────────────────────────────────────────
def plot_factor_bars(shap_vals, X_patient, n=7):
    sv = pd.Series(shap_vals, index=FEATURE_COLS)
    top_idx = sv.abs().nlargest(n).index.tolist()
    vals    = [sv[f] for f in top_idx]
    labels  = [CLINICAL_LABELS.get(f, f.replace("_", " ").replace("  ", " ").title())
               for f in top_idx]
    raw_vals = [X_patient[f].values[0] for f in top_idx]

    colors = ["#d64045" if v > 0 else "#1a7a8a" for v in vals]

    fig, ax = plt.subplots(figsize=(7.5, 0.58 * n + 0.9))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    bars = ax.barh(range(n), vals, color=colors, alpha=0.82, height=0.55, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.axvline(0, color="#444444", lw=1.1, zorder=3)

    # valor de la variable junto a cada barra
    x_max = max(abs(v) for v in vals) if vals else 1
    for i, (v, rv) in enumerate(zip(vals, raw_vals)):
        txt = f" {rv}" if v >= 0 else f"{rv} "
        ha  = "left" if v >= 0 else "right"
        ax.text(v, i, txt, ha=ha, va="center", fontsize=9, color="#333333")

    ax.set_xlim(-x_max * 1.45, x_max * 1.45)
    ax.set_xlabel("← Reduce el riesgo          Aumenta el riesgo →",
                  fontsize=9, color="#666666")
    ax.tick_params(axis="x", labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    plt.tight_layout(pad=0.4)
    return fig


# ── bloque de recomendaciones ─────────────────────────────────────────────────
def render_recommendations(tier):
    r = RECOMMENDATIONS[tier]
    dot_color = r["border"]
    items_html = "".join(
        f"""<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">
              <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};
                          margin-top:5px;flex-shrink:0;"></div>
              <span style="font-size:13px;line-height:1.5;opacity:0.85;">{a}</span>
            </div>"""
        for a in r["acciones"]
    )
    st.markdown(
        f"""
        <div style="border:1px solid #D8D6D0; border-radius:3px;
                    padding:18px 20px; margin-top:4px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
            <div style="width:10px;height:10px;border-radius:50%;background:{dot_color};"></div>
            <span style="font-size:13px;font-weight:600;letter-spacing:0.02em;">{r['titulo']}</span>
          </div>
          {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ReAdmit · Predicción de reingreso",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; }

/* Topbar */
.topbar {
    display: flex; align-items: baseline; gap: 16px;
    padding: 0 0 16px 0; margin-bottom: 24px;
    border-bottom: 2px solid #0F172A;
}
.topbar-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px; font-weight: 500; color: #0D9488;
}
.topbar-sub { font-size: 12px; opacity: 0.5; }

/* Separadores sidebar */
.sidebar-section {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; margin: 16px 0 4px 0;
    opacity: 0.75;
}

/* Tarjeta paciente */
.patient-card {
    border: 1px solid #D8D6D0;
    border-radius: 3px; padding: 14px 16px;
    font-size: 13px;
}

/* Badge modelo */
.model-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; opacity: 0.8;
    display: inline-block; margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

model, is_calibrated = load_model()

# ── cabecera ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <span class="topbar-name">ReAdmit</span>
  <span class="topbar-sub">predicción de reingreso a 30 días &nbsp;/&nbsp; apoyo a la decisión al alta</span>
</div>
""", unsafe_allow_html=True)

if not is_calibrated:
    st.warning("Modelo sin calibración cargado (fallback). Las probabilidades pueden estar sobreestimadas.")

# ── sidebar: formulario clínico ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Datos del paciente al alta")

    st.markdown('<p class="sidebar-section">Datos clínicos</p>', unsafe_allow_html=True)
    age      = st.number_input("Edad al ingreso (años)", 18, 110, 65, step=1)
    gender   = st.radio("Sexo", ["Femenino", "Masculino"], horizontal=True)
    los      = st.number_input("Días de estancia", 0, 365, 5, step=1,
                               help="Número de días desde el ingreso hasta el alta")
    prev_adm = st.number_input("Ingresos previos (total histórico)", 0, 50, 0, step=1,
                               help="Total de ingresos hospitalarios previos del paciente en MIMIC")
    n_diag   = st.number_input("Número de diagnósticos al alta", 1, 50, 5, step=1,
                               help="Total de diagnósticos registrados en este episodio")

    st.markdown('<p class="sidebar-section">Datos del ingreso</p>', unsafe_allow_html=True)
    admission_type = st.selectbox("Tipo de admisión", [
        "EW EMER.", "DIRECT EMER.", "ELECTIVE", "URGENT",
        "SURGICAL SAME DAY ADMISSION", "DIRECT OBSERVATION",
        "EU OBSERVATION", "OBSERVATION ADMIT",
    ])
    admission_loc = st.selectbox("Procedencia al ingreso", [
        "EMERGENCY ROOM", "PHYSICIAN REFERRAL", "CLINIC REFERRAL",
        "TRANSFER FROM HOSPITAL", "TRANSFER FROM SKILLED NURSING FACILITY",
        "WALK-IN/SELF REFERRAL", "PROCEDURE SITE", "PACU",
        "INFORMATION NOT AVAILABLE", "INTERNAL TRANSFER TO OR FROM PSYCH",
    ])
    insurance = st.selectbox("Tipo de cobertura", [
        "Medicaid", "Medicare", "Private", "No charge", "Other", "Unknown",
    ])

    st.markdown('<p class="sidebar-section">Datos del alta</p>', unsafe_allow_html=True)
    discharge_loc = st.selectbox("Destino al alta", [
        "HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY",
        "REHAB", "Other", "Unknown",
    ])
    marital  = st.selectbox("Estado civil", ["SINGLE", "MARRIED", "WIDOWED", "Unknown"])
    language = st.selectbox("Idioma principal", ["English", "Spanish", "Russian", "Other"])

    with st.expander("Datos sociodemográficos adicionales"):
        race = st.selectbox("Raza/etnia", [
            "WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC OR LATINO",
            "ASIAN - CHINESE", "OTHER", "UNKNOWN", "Other",
            "BLACK/CAPE VERDEAN", "HISPANIC/LATINO - DOMINICAN",
            "HISPANIC/LATINO - PUERTO RICAN", "WHITE - OTHER EUROPEAN", "WHITE - RUSSIAN",
        ])

    st.divider()
    predict_btn = st.button("Calcular riesgo de reingreso",
                            type="primary", use_container_width=True)

    st.markdown(
        '<div class="model-badge">LightGBM · AUC 0,657 · ECE 0,007 · MIMIC-IV</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PANEL PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
if predict_btn:
    race_val = race

    X_patient = build_features(
        age, los, prev_adm, n_diag, gender,
        race_val, insurance, marital, language,
        admission_type, admission_loc, discharge_loc,
    )

    prob = model.predict_proba(X_patient)[0, 1]

    if prob < 0.25:
        tier, tier_color = "BAJO",     "#28a745"
    elif prob < 0.40:
        tier, tier_color = "MODERADO", "#856404"
    else:
        tier, tier_color = "ALTO",     "#dc3545"

    # ── fila 1: medidor + resumen del paciente ────────────────────────────────
    col_gauge, col_patient = st.columns([1, 1], gap="large")

    with col_gauge:
        st.subheader("Resultado de la evaluación")
        fig_gauge = plot_gauge(prob)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        # Barra de contexto poblacional
        st.caption(
            f"Referencia: prevalencia de reingreso en el conjunto de evaluación = **34,6 %**. "
            f"El umbral de alerta habitual se sitúa en el **30–40 %**."
        )

    with col_patient:
        st.subheader("Resumen del paciente")

        discharge_labels = {
            "HOME": "Domicilio", "HOME HEALTH CARE": "Atención domiciliaria",
            "SKILLED NURSING FACILITY": "Residencia/Enfermería",
            "REHAB": "Rehabilitación", "Other": "Otro", "Unknown": "Desconocido",
        }
        adm_labels = {
            "EW EMER.": "Urgencias (emergencia)", "DIRECT EMER.": "Emergencia directa",
            "ELECTIVE": "Programado", "URGENT": "Urgente",
            "SURGICAL SAME DAY ADMISSION": "Cirugía ambulatoria",
            "DIRECT OBSERVATION": "Observación directa",
            "EU OBSERVATION": "Observación urgencias", "OBSERVATION ADMIT": "Observación",
        }

        def field(label, val):
            return f"""<div style="margin-bottom:10px;">
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;opacity:0.7;margin-bottom:2px;">{label}</div>
              <div style="font-size:13px;font-weight:600;">{val}</div>
            </div>"""

        st.markdown(
            f"""
            <div class="patient-card">
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.1em;
                          color:#e05c3a;font-weight:600;margin-bottom:14px;
                          font-family:'IBM Plex Mono',monospace;">
                Ficha del episodio
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 20px;">
                {field("Edad / Sexo", f"{age} años · {gender}")}
                {field("Estancia", f"{los} día{'s' if los != 1 else ''}")}
                {field("Diagnósticos", n_diag)}
                {field("Ingresos previos", prev_adm)}
                {field("Tipo de admisión", adm_labels.get(admission_type, admission_type))}
                {field("Destino al alta", discharge_labels.get(discharge_loc, discharge_loc))}
                {field("Cobertura", insurance)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:{tier_color}; border-radius:8px; padding:18px 22px; margin-top:12px;">
              <div style="font-family:'IBM Plex Mono',monospace; font-size:10px;
                          color:rgba(255,255,255,0.7); letter-spacing:0.12em; text-transform:uppercase;
                          margin-bottom:8px;">
                Nivel de riesgo estimado
              </div>
              <div style="font-size:40px; font-weight:700; color:#fff;
                          letter-spacing:-0.02em; line-height:1;">
                {tier}
              </div>
              <div style="font-size:14px; color:rgba(255,255,255,0.85); margin-top:8px; font-weight:500;">
                probabilidad estimada: {prob:.1%}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── fila 2: factores + recomendaciones ────────────────────────────────────
    col_factors, col_recs = st.columns([1.1, 1], gap="large")

    with col_factors:
        st.subheader("¿Qué factores influyen en este resultado?")
        st.caption(
            "Las barras rojas aumentan el riesgo estimado; "
            "las azules lo reducen. El valor de la variable de este paciente "
            "aparece junto a cada barra."
        )
        with st.spinner("Calculando…"):
            base_lgbm = model.base_model if hasattr(model, "base_model") else model
            explainer  = shap.TreeExplainer(base_lgbm)
            sv_obj     = explainer(X_patient)

        fig_bars = plot_factor_bars(sv_obj[0].values, X_patient, n=7)
        st.pyplot(fig_bars, use_container_width=True)
        plt.close(fig_bars)

    with col_recs:
        st.subheader("Recomendaciones clínicas")
        render_recommendations(tier)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interpretación rápida de los 3 factores más influyentes en texto
        sv_series = pd.Series(sv_obj[0].values, index=FEATURE_COLS)
        top3 = sv_series.abs().nlargest(3).index.tolist()

        lines = []
        for feat in top3:
            lbl = CLINICAL_LABELS.get(feat, feat.replace("_", " ").title())
            val = X_patient[feat].values[0]
            direction = "↑ aumenta" if sv_series[feat] > 0 else "↓ reduce"
            lines.append(f"- **{lbl}** ({val}) — {direction} el riesgo")

        st.markdown(
            "**Los 3 factores más determinantes para este paciente:**\n\n"
            + "\n".join(lines)
        )

    # ── detalles técnicos (colapsado) ─────────────────────────────────────────
    with st.expander("Gráfico SHAP detallado (referencia técnica)"):
        shap.plots.waterfall(sv_obj[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close("all")

    st.markdown("---")
    st.caption(
        "ReAdmit · Prototipo académico — TFM UTAMED 2025–2026. "
        "Modelo entrenado con datos de MIMIC-IV (Beth Israel Deaconess Medical Center, Boston, 2008–2019). "
        "Requiere validación externa y recalibración local antes de cualquier uso clínico real. "
        "No sustituye el juicio clínico del profesional sanitario."
    )

else:
    # ── pantalla de bienvenida ────────────────────────────────────────────────
    st.markdown("""
    <div style="max-width:680px; margin: 40px auto 0 auto;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#C44D34;
                  letter-spacing:0.1em; text-transform:uppercase; margin-bottom:14px;">
        Prueba de concepto · TFM UTAMED 2025–2026
      </div>
      <div style="font-size:28px; font-weight:600; line-height:1.25; margin-bottom:14px;">
        Predicción de reingreso hospitalario<br>en los 30 días siguientes al alta
      </div>
      <div style="font-size:14px; opacity:0.85; line-height:1.7; margin-bottom:36px;">
        Introduce los datos del paciente en el panel lateral y obtén la estimación
        de riesgo, los factores que más han influido en esa predicción concreta
        y las acciones clínicas recomendadas.
      </div>

      <div style="display:flex; gap:40px; margin-bottom:36px; border-top:1px solid #CBD5E1; padding-top:20px;">
        <div>
          <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.09em; opacity:0.7; margin-bottom:4px;">Algoritmo</div>
          <div style="font-size:14px; font-weight:600;">LightGBM + calibración isotónica</div>
        </div>
        <div>
          <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.09em; opacity:0.7; margin-bottom:4px;">ROC-AUC (test)</div>
          <div style="font-size:14px; font-weight:600;">0,657 <span style="opacity:0.6; font-weight:400; font-size:12px;">· ref. 0,62–0,72</span></div>
        </div>
        <div>
          <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.09em; opacity:0.7; margin-bottom:4px;">Datos de entrenamiento</div>
          <div style="font-size:14px; font-weight:600;">MIMIC-IV · 315.982 ingresos</div>
        </div>
      </div>

      <div style="font-size:12px; opacity:0.65; border-top:1px solid #D8D6D0; padding-top:12px;">
        Variables: datos administrativos del HIS al alta (estancia, diagnósticos, tipo de admisión,
        destino al alta, cobertura, datos sociodemográficos). Sin registros clínicos adicionales.
      </div>
    </div>
    """, unsafe_allow_html=True)
