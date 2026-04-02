"""
Heart Disease Detection — Streamlit UI
Run: streamlit run app.py
"""

import subprocess, pathlib
if not pathlib.Path("final_model.pkl").exists():
    subprocess.run(["python", "train.py"], check=True)

import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import pandas as pd






# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Detector",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
  }

  /* Header banner */
  .hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "🫀";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.15;
  }
  .hero h1 { color: white !important; margin: 0; font-size: 2.4rem; }
  .hero p  { color: #a0b4cc; margin: 0.5rem 0 0; font-size: 1rem; }

  /* Section cards */
  .section-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .section-title {
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1rem;
  }

  /* Result boxes */
  .result-high {
    background: linear-gradient(135deg, #fff5f5, #fed7d7);
    border: 2px solid #fc8181;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse 1.5s ease-in-out 3;
  }
  .result-low {
    background: linear-gradient(135deg, #f0fff4, #c6f6d5);
    border: 2px solid #68d391;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }
  .result-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
  .result-sub   { font-size: 0.95rem; color: #555; margin-top: 0.5rem; }

  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(252,129,129,0.4); }
    50%      { box-shadow: 0 0 0 12px rgba(252,129,129,0); }
  }

  /* Metric badges */
  .metric-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1rem; }
  .metric-badge {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
  }
  .metric-badge strong { display: block; font-size: 1.2rem; color: #1e40af; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0f172a; }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Disclaimer */
  .disclaimer {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #78350f;
    margin-top: 1.5rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = Path("final_model.pkl")
    if not path.exists():
        return None
    return joblib.load(path)

model = load_model()

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Heart Disease Risk Predictor</h1>
  <p>XGBoost-powered clinical screening tool &nbsp;·&nbsp; CN6000 Final Year Project</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️  **Model not found.**  Run `python train.py` first to generate `final_model.pkl`.")
    st.stop()

# ── Sidebar — about ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 About")
    st.markdown("""
This tool uses an **XGBoost classifier** trained on the UCI Heart Disease dataset (1,025 samples).

**Features used:**
- Demographic: Age, Sex
- Vital signs: Resting BP, Cholesterol, Max HR
- Clinical tests: ECG, Exercise angina, ST depression, Thalassemia

**Model performance:**
- Accuracy ≈ 98%
- ROC-AUC ≈ 0.99

---
⚠️ For educational use only. Not a medical device.
""")
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repository](https://github.com)")
    st.markdown("[UCI Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### Patient Information")
st.markdown("Fill in the clinical measurements below and click **Predict**.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
    age   = st.slider("Age (years)", 20, 80, 50)
    sex   = st.radio("Sex", ["Female", "Male"], horizontal=True)
    sex_v = 1 if sex == "Male" else 0

    st.markdown('<div class="section-title" style="margin-top:1.2rem;">🩺 Chest Pain Type</div>', unsafe_allow_html=True)
    cp_map = {
        "Typical Angina (0)": 0,
        "Atypical Angina (1)": 1,
        "Non-Anginal Pain (2)": 2,
        "Asymptomatic (3)": 3
    }
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    cp_v = cp_map[cp]

with col2:
    st.markdown('<div class="section-title">💉 Vital Signs</div>', unsafe_allow_html=True)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol     = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)

    fbs_map = {"≤ 120 mg/dl (0)": 0, "> 120 mg/dl (1)": 1}
    fbs   = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()))
    fbs_v = fbs_map[fbs]

with col3:
    st.markdown('<div class="section-title">📊 Clinical Tests</div>', unsafe_allow_html=True)

    ecg_map = {"Normal (0)": 0, "ST-T Abnormality (1)": 1, "LV Hypertrophy (2)": 2}
    restecg   = st.selectbox("Resting ECG Results", list(ecg_map.keys()))
    restecg_v = ecg_map[restecg]

    exang_map = {"No (0)": 0, "Yes (1)": 1}
    exang   = st.selectbox("Exercise-Induced Angina", list(exang_map.keys()))
    exang_v = exang_map[exang]

    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)

    slope_map = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
    slope   = st.selectbox("Slope of ST Segment", list(slope_map.keys()))
    slope_v = slope_map[slope]

    ca   = st.selectbox("Major Vessels Coloured (0–4)", [0, 1, 2, 3, 4])

    thal_map = {"Normal (1)": 1, "Fixed Defect (2)": 2, "Reversible Defect (3)": 3}
    thal   = st.selectbox("Thalassemia Type", list(thal_map.keys()))
    thal_v = thal_map[thal]

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("---")
predict_col, _, info_col = st.columns([1, 0.2, 2])

with predict_col:
    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

if predict_btn:
    input_data = np.array([[
        age, sex_v, cp_v, trestbps, chol,
        fbs_v, restecg_v, thalach, exang_v,
        oldpeak, slope_v, ca, thal_v
    ]])

    prediction   = model.predict(input_data)[0]
    probability  = model.predict_proba(input_data)[0]
    risk_pct     = probability[1] * 100
    healthy_pct  = probability[0] * 100

    with info_col:
        if prediction == 1:
            st.markdown(f"""
<div class="result-high">
  <div class="result-title">⚠️ Elevated Risk Detected</div>
  <div class="result-sub">Model confidence: <strong>{risk_pct:.1f}%</strong> probability of heart disease</div>
  <div class="metric-row">
    <div class="metric-badge"><strong>{risk_pct:.1f}%</strong>Disease Risk</div>
    <div class="metric-badge"><strong>{healthy_pct:.1f}%</strong>Healthy</div>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-low">
  <div class="result-title">✅ Low Risk</div>
  <div class="result-sub">Model confidence: <strong>{healthy_pct:.1f}%</strong> probability of no disease</div>
  <div class="metric-row">
    <div class="metric-badge"><strong>{healthy_pct:.1f}%</strong>Healthy</div>
    <div class="metric-badge"><strong>{risk_pct:.1f}%</strong>Disease Risk</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Show input summary
        with st.expander("📋 Input Summary"):
            summary_df = pd.DataFrame({
                "Feature": ["Age","Sex","Chest Pain","Resting BP","Cholesterol",
                             "Fasting BS","ECG","Max HR","Ex. Angina",
                             "ST Depression","ST Slope","Vessels","Thalassemia"],
                "Value": [age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
  It is not a substitute for professional medical advice, diagnosis, or treatment.
  Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

# ── Model plots (if available) ─────────────────────────────────────────────────
if Path("plots").exists():
    st.markdown("---")
    st.markdown("### 📈 Model Performance Plots")
    plot_cols = st.columns(4)
    plot_files = [
        ("plots/confusion_matrix.png", "Confusion Matrix"),
        ("plots/roc_curve.png", "ROC Curve"),
        ("plots/target_distribution.png", "Class Distribution"),
        ("plots/age_distribution.png", "Age by Class"),
    ]
    for (path, caption), col in zip(plot_files, plot_cols):
        if Path(path).exists():
            with col:
                st.image(path, caption=caption, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>"
    "Heart Disease Risk Predictor · CN6000 Final Year Project · Built with Streamlit & XGBoost"
    "</div>",
    unsafe_allow_html=True
)
