"""
Mommy Care — AI Pregnancy Monitoring App
Streamlit frontend with integrated ML backend
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import time

from utils.ml_engine import load_bundle, predict_risk, get_feature_importance, train_all_models
from utils.helpers import (
    calculate_hpl, calculate_financial_plan,
    generate_recommendations, TRIMESTER_CONTENT
)

# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mommy Care — AI Pregnancy Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp { background-color: #F7F5F2; }

/* Hide default streamlit header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1E2D40 !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #000000 !important; font-size: 12px !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: white !important;
}
[data-testid="stSidebar"] .stSlider label { font-size: 12px !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid rgba(30,45,64,0.08);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    box-shadow: none;
}

/* Buttons */
.stButton > button {
    background: #E8547A !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 14px !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Card-like containers */
.info-card {
    background: white;
    border-radius: 14px;
    border: 1px solid rgba(30,45,64,0.08);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.hpl-card {
    background: linear-gradient(135deg, #E8547A 0%, #C0406A 100%);
    border-radius: 16px;
    padding: 1.5rem;
    color: white;
    margin-bottom: 1rem;
}
.hpl-title { font-size: 11px; font-weight: 600; opacity: .7; letter-spacing: .8px; margin-bottom: 4px; }
.hpl-date { font-family: 'DM Serif Display', serif; font-size: 28px; margin-bottom: 4px; }
.hpl-sub { font-size: 13px; opacity: .8; }

.badge-low { background:#E8F8F4; color:#0F6E56; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-mid { background:#FEF3C7; color:#92400E; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-high { background:#FEE2E2; color:#991B1B; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }

.rec-high { border-left: 4px solid #E8547A; background: #FDF0F4; }
.rec-mid  { border-left: 4px solid #F59E0B; background: #FFFBEB; }
.rec-low  { border-left: 4px solid #1DAB87; background: #E8F8F4; }

.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .8px;
    color: #4B6280;
    text-transform: uppercase;
    margin-bottom: .5rem;
}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { border-radius: 14px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────────────────────────
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "hpl_data" not in st.session_state:
    st.session_state.hpl_data = None
if "input_data" not in st.session_state:
    st.session_state.input_data = {}


# ── Ensure models are trained ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle()


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
      <div style="font-size:40px;margin-bottom:8px"></div>
      <div style="font-family:'DM Serif Display',serif;font-size:22px;color:white">Mommy Care</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:4px">AI Pregnancy Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###  Data Kehamilan")
    hpht_input = st.date_input(
        "Tanggal pertama menstruasi terakhir (HPHT)",
        value=date(2024, 10, 1),
        max_value=date.today(),
        format="DD/MM/YYYY",
    )

    st.markdown("---")
    st.markdown("###  Data Kesehatan Ibu")

    age = st.slider("Usia Ibu (tahun)", 17, 48, 28)
    systolic_bp = st.slider("Tekanan Darah Sistolik (mmHg)", 90, 180, 115)
    diastolic_bp = st.slider("Tekanan Darah Diastolik (mmHg)", 60, 120, 75)
    blood_glucose = st.slider("Gula Darah (mg/dL)", 70, 200, 90)
    body_temp = st.slider("Suhu Tubuh (°C)", 36.0, 39.5, 36.6, step=0.1)
    heart_rate = st.slider("Detak Jantung (bpm)", 55, 120, 76)
    weight_gain = st.slider("Kenaikan Berat Badan (kg)", 0, 22, 8)

    st.markdown("---")
    st.markdown("###  Riwayat")
    prev_preg = st.selectbox("Jumlah Kehamilan Sebelumnya", [0, 1, 2, 3, 4], index=0)
    prev_comp = st.selectbox("Riwayat Komplikasi?", ["Tidak", "Ya"], index=0)

    st.markdown("---")
    st.markdown("### Pilih Model AI")
    model_choice = st.selectbox(
        "Algoritma ML",
        ["Random Forest", "Logistic Regression", "Gradient Boosting"],
        index=0
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button(" Analisis Risiko Sekarang", use_container_width=True)


# ── Process Input ───────────────────────────────────────────────────────────────
hpl_data = calculate_hpl(datetime.combine(hpht_input, datetime.min.time()))
st.session_state.hpl_data = hpl_data

input_data = {
    "age": age,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "blood_glucose": blood_glucose,
    "body_temp": body_temp,
    "heart_rate": heart_rate,
    "weight_gain_kg": weight_gain,
    "gestational_age_weeks": hpl_data["gestational_weeks"],
    "previous_pregnancies": prev_preg,
    "previous_complications": 1 if prev_comp == "Ya" else 0,
}
st.session_state.input_data = input_data

if analyze_btn:
    with st.spinner("Model AI sedang menganalisis data kesehatan Anda..."):
        bundle = get_bundle()
        pred = predict_risk(input_data, model_name=model_choice)
        st.session_state.prediction = pred
        st.session_state.model_used = model_choice


# ── Auto-run prediction on first load ──────────────────────────────────────────
if st.session_state.prediction is None:
    bundle = get_bundle()
    pred = predict_risk(input_data, model_name=model_choice)
    st.session_state.prediction = pred
    st.session_state.model_used = model_choice


# ── MAIN CONTENT ────────────────────────────────────────────────────────────────
prediction = st.session_state.prediction
fin_plan = calculate_financial_plan(
    prediction["risk_level"],
    prediction["cs_risk"],
    hpl_data["gestational_weeks"]
)

# Page header
st.markdown("""
<div style="margin-bottom:1.5rem">
  <h1 style="font-family:'DM Serif Display',serif;font-size:32px;color:#1E2D40;margin:0">
    Dashboard Pemantauan Kehamilan
  </h1>
  <p style="color:#4B6280;font-size:14px;margin-top:4px">
    Powered by AI · Model: """ + st.session_state.get("model_used", model_choice) + """
  </p>
</div>
""", unsafe_allow_html=True)


# ── Row 1: HPL + Metrics ────────────────────────────────────────────────────────
col_hpl, col_m1, col_m2, col_m3 = st.columns([1.8, 1, 1, 1])

with col_hpl:
    hpl_str = hpl_data["hpl"].strftime("%d %B %Y")
    gest_str = f"{hpl_data['gestational_weeks']} minggu {hpl_data['gestational_days_rem']} hari"
    st.markdown(f"""
    <div class="hpl-card">
      <div class="hpl-title">HARI PERKIRAAN LAHIR (NAEGELE'S RULE)</div>
      <div class="hpl-date">{hpl_str}</div>
      <div class="hpl-sub">Usia kandungan: {gest_str} · {hpl_data['trimester_label']}</div>
      <div style="margin-top:12px;background:rgba(255,255,255,0.15);border-radius:20px;height:8px;overflow:hidden">
        <div style="height:100%;width:{hpl_data['progress_pct']}%;background:rgba(255,255,255,0.8);border-radius:20px;transition:width .8s ease"></div>
      </div>
      <div style="font-size:11px;opacity:.75;margin-top:5px">{hpl_data['progress_pct']:.0f}% perjalanan kehamilan · {hpl_data['days_to_hpl']} hari lagi</div>
    </div>
    """, unsafe_allow_html=True)

with col_m1:
    risk_map = {"low": "🟢 Rendah", "mid": "🟡 Sedang", "high": "🔴 Tinggi"}
    risk_delta_map = {"low": None, "mid": "Perlu pemantauan", "high": "Butuh perhatian segera"}
    st.metric(
        "Tingkat Risiko",
        risk_map[prediction["risk_level"]],
        delta=risk_delta_map[prediction["risk_level"]],
        delta_color="inverse" if prediction["risk_level"] != "low" else "normal"
    )

with col_m2:
    st.metric(
        "Risiko Preeklamsia",
        f"{prediction['preeclampsia_risk']*100:.0f}%",
        delta="tinggi" if prediction['preeclampsia_risk'] > 0.4 else None,
        delta_color="inverse"
    )

with col_m3:
    st.metric(
        "Kemungkinan Caesar",
        f"{prediction['cs_risk']*100:.0f}%",
        delta="tinggi" if prediction['cs_risk'] > 0.5 else None,
        delta_color="inverse"
    )


# ── Row 2: Risk Gauge + Probability + Financial ─────────────────────────────────
col_gauge, col_prob, col_fin = st.columns([1.2, 1, 1.2])

with col_gauge:
    risk_pct = (
        prediction["probabilities"]["low"] * 10 +
        prediction["probabilities"]["mid"] * 50 +
        prediction["probabilities"]["high"] * 100
    )
    color_val = "#1DAB87" if risk_pct < 30 else "#F59E0B" if risk_pct < 60 else "#E8547A"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(risk_pct, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Skor Risiko Kehamilan", "font": {"size": 14, "family": "DM Sans", "color": "#4B6280"}},
        number={"suffix": "%", "font": {"size": 36, "family": "DM Sans", "color": "#1E2D40"}},
        delta={"reference": 30, "position": "bottom"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1", "nticks": 6},
            "bar": {"color": color_val, "thickness": 0.35},
            "bgcolor": "#F7F5F2",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#E8F8F4"},
                {"range": [33, 66], "color": "#FEF3C7"},
                {"range": [66, 100], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "#1E2D40", "width": 2},
                "thickness": 0.8,
                "value": risk_pct,
            },
        }
    ))
    fig_gauge.update_layout(
        height=280, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="white", plot_bgcolor="white",
        font_family="DM Sans",
    )
    with st.container():
        st.markdown('<div class="info-card" style="padding:.5rem">', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

with col_prob:
    probs = prediction["probabilities"]
    labels = ["Rendah", "Sedang", "Tinggi"]
    values = [probs["low"], probs["mid"], probs["high"]]
    colors_pie = ["#1DAB87", "#F59E0B", "#E8547A"]

    fig_pie = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55, pull=[0.04, 0, 0],
        marker=dict(colors=colors_pie, line=dict(color="white", width=2)),
        textinfo="percent", textfont=dict(family="DM Sans", size=13),
        showlegend=True,
    ))
    fig_pie.update_layout(
        title={"text": "Distribusi Probabilitas", "font": {"size": 14, "family": "DM Sans", "color": "#4B6280"}},
        height=280, margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(font=dict(family="DM Sans", size=12), orientation="h", y=-0.05),
        font_family="DM Sans",
        annotations=[dict(
            text=f"<b>{prediction['risk_level'].upper()}</b>",
            x=0.5, y=0.5, font_size=14, font_family="DM Sans",
            showarrow=False, font_color="#1E2D40"
        )],
    )
    with st.container():
        st.markdown('<div class="info-card" style="padding:.5rem">', unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

with col_fin:
    st.markdown(f"""
    <div class="info-card">
      <div class="section-label"> Perencanaan Finansial Persalinan</div>
      <div style="margin-top:.75rem">
        <div style="display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid rgba(30,45,64,0.07);font-size:13px">
          <span style="color:#4B6280">Jenis rencana</span>
          <span style="font-weight:500;color:#1E2D40">{fin_plan['plan_type']}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid rgba(30,45,64,0.07);font-size:13px">
          <span style="color:#4B6280">Biaya normal (est.)</span>
          <span style="font-weight:500;color:#1E2D40">Rp {fin_plan['normal_cost']:,.0f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid rgba(30,45,64,0.07);font-size:13px">
          <span style="color:#4B6280">Biaya caesar (est.)</span>
          <span style="font-weight:500;color:#1E2D40">Rp {fin_plan['caesar_cost']:,.0f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:.6rem 0;border-bottom:1px solid rgba(30,45,64,0.07);font-size:14px">
          <span style="color:#4B6280;font-weight:500">Target tabungan</span>
          <span style="font-weight:600;color:#E8547A">Rp {fin_plan['target']:,.0f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:.6rem 0;font-size:14px">
          <span style="color:#4B6280;font-weight:500">Tabungan/bulan</span>
          <span style="font-weight:600;color:#E8547A">Rp {fin_plan['monthly_savings']:,.0f}</span>
        </div>
      </div>
      {"<div style='background:#FEF3C7;border-radius:8px;padding:.5rem .75rem;font-size:12px;color:#92400E;margin-top:.5rem'>⚠️ Disarankan mengambil asuransi persalinan.</div>" if fin_plan['insurance_rec'] else ""}
      <div style="font-size:11px;color:#94a3b8;margin-top:.5rem">Sisa waktu ≈ {fin_plan['months_remaining']:.0f} bulan</div>
    </div>
    """, unsafe_allow_html=True)


# ── Row 3: Sub-risk Bars + Feature Importance ───────────────────────────────────
col_sub, col_fi = st.columns([1, 1])

with col_sub:
    risks = {
        "Preeklamsia": prediction["preeclampsia_risk"],
        "Diabetes Gestasional": prediction["gd_risk"],
        "Persalinan Caesar": prediction["cs_risk"],
    }
    r_colors = ["#E8547A" if v >= 0.5 else "#F59E0B" if v >= 0.25 else "#1DAB87" for v in risks.values()]

    fig_bar = go.Figure(go.Bar(
        x=list(risks.values()),
        y=list(risks.keys()),
        orientation='h',
        marker=dict(color=r_colors, line=dict(width=0)),
        text=[f"{v*100:.0f}%" for v in risks.values()],
        textposition="outside",
        textfont=dict(family="DM Sans", size=13),
    ))
    fig_bar.update_layout(
        title={"text": "Risiko Komplikasi Spesifik", "font": {"size": 14, "family": "DM Sans", "color": "#4B6280"}},
        xaxis=dict(range=[0, 1.1], tickformat=".0%", showgrid=True, gridcolor="#F0EDE8"),
        yaxis=dict(tickfont=dict(family="DM Sans", size=13)),
        height=240, margin=dict(t=40, b=20, l=10, r=60),
        paper_bgcolor="white", plot_bgcolor="white",
        font_family="DM Sans",
    )
    with st.container():
        st.markdown('<div class="info-card" style="padding:.5rem">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

with col_fi:
    fi = get_feature_importance(st.session_state.get("model_used", model_choice))
    if fi:
        fi_labels = {
            'age': 'Usia', 'systolic_bp': 'Sistolik', 'diastolic_bp': 'Diastolik',
            'blood_glucose': 'Gula Darah', 'body_temp': 'Suhu Tubuh',
            'heart_rate': 'Detak Jantung', 'weight_gain_kg': 'Kenaikan BB',
            'gestational_age_weeks': 'Usia Kandungan',
            'previous_pregnancies': 'Kehamilan Sebelumnya',
            'previous_complications': 'Riwayat Komplikasi'
        }
        top_fi = dict(list(fi.items())[:7])
        fig_fi = go.Figure(go.Bar(
            x=list(top_fi.values()),
            y=[fi_labels.get(k, k) for k in top_fi.keys()],
            orientation='h',
            marker=dict(color="#C0406A", opacity=0.75, line=dict(width=0)),
            text=[f"{v:.3f}" for v in top_fi.values()],
            textposition="outside",
            textfont=dict(family="DM Sans", size=11),
        ))
        fig_fi.update_layout(
            title={"text": f"Fitur Penting — {st.session_state.get('model_used', model_choice)}", "font": {"size": 14, "family": "DM Sans", "color": "#4B6280"}},
            xaxis=dict(showgrid=True, gridcolor="#F0EDE8"),
            yaxis=dict(tickfont=dict(family="DM Sans", size=12)),
            height=240, margin=dict(t=40, b=20, l=10, r=60),
            paper_bgcolor="white", plot_bgcolor="white",
            font_family="DM Sans",
        )
        with st.container():
            st.markdown('<div class="info-card" style="padding:.5rem">', unsafe_allow_html=True)
            st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Feature importance tidak tersedia untuk model ini.")


# ── Row 4: Recommendations ──────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:.5rem">Rekomendasi AI Personalisasi</div>', unsafe_allow_html=True)

recs = generate_recommendations(input_data, prediction)
cols_rec = st.columns(min(len(recs), 3))
for i, rec in enumerate(recs):
    urgency_class = f"rec-{rec['urgency']}"
    with cols_rec[i % 3]:
        st.markdown(f"""
        <div class="info-card {urgency_class}" style="min-height:110px">
          <div style="font-size:18px;margin-bottom:6px">{rec['icon']} <span style="font-size:11px;color:#6B7280;font-weight:600;letter-spacing:.5px">{rec['category'].upper()}</span></div>
          <div style="font-size:13px;font-weight:600;color:#1E2D40;margin-bottom:4px">{rec['title']}</div>
          <div style="font-size:12px;color:#4B6280;line-height:1.6">{rec['detail']}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Row 5: Trimester Education ──────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:.5rem">Edukasi Trimester</div>', unsafe_allow_html=True)

current_trim = hpl_data["trimester"]
tab1, tab2, tab3 = st.tabs(["🌱 Trimester 1", "🌿 Trimester 2", "🌸 Trimester 3"])

for tab, trim_num in zip([tab1, tab2, tab3], [1, 2, 3]):
    with tab:
        content = TRIMESTER_CONTENT[trim_num]
        is_current = (trim_num == current_trim)
        if is_current:
            st.success(f"✨ Anda sedang berada di **{content['title']}**")
        cols_edu = st.columns(2)
        for i, (icon, title, detail) in enumerate(content["items"]):
            with cols_edu[i % 2]:
                st.markdown(f"""
                <div style="background:white;border-radius:12px;border:1px solid rgba(30,45,64,0.07);padding:1rem;margin-bottom:.5rem">
                  <div style="font-size:20px;margin-bottom:5px">{icon}</div>
                  <div style="font-size:13px;font-weight:600;color:#1E2D40;margin-bottom:3px">{title}</div>
                  <div style="font-size:12px;color:#4B6280;line-height:1.5">{detail}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Row 6: Model Performance ────────────────────────────────────────────────────
with st.expander("📊 Performa Model ML", expanded=False):
    bundle = get_bundle()
    metrics = bundle["metrics"]

    perf_data = []
    for model_name, m in metrics.items():
        perf_data.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']*100:.1f}%",
            "CV Mean": f"{m['cv_mean']*100:.1f}%",
            "CV Std": f"± {m['cv_std']*100:.1f}%",
        })
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.markdown("**Confusion Matrix — " + st.session_state.get("model_used", model_choice) + "**")
    cm = metrics[st.session_state.get("model_used", model_choice)]["confusion_matrix"]
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Rendah", "Sedang", "Tinggi"],
        y=["Rendah", "Sedang", "Tinggi"],
        color_continuous_scale=[[0, "#F7F5F2"], [0.5, "#F5B8C8"], [1, "#E8547A"]],
        labels=dict(x="Predicted", y="Actual"),
    )
    fig_cm.update_layout(
        height=300, margin=dict(t=20, b=20, l=10, r=10),
        paper_bgcolor="white", font_family="DM Sans",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    col_retrain, _ = st.columns([1, 3])
    with col_retrain:
        if st.button("🔄 Retrain Semua Model"):
            with st.spinner("Training ulang semua model..."):
                bundle = train_all_models()
                st.cache_resource.clear()
            st.success("Model berhasil di-retrain!")
            st.rerun()


# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-size:12px;color:#94a3b8">
  Mommy Care — Team Anomali · POC Demo · Bukan pengganti konsultasi medis profesional
</div>
""", unsafe_allow_html=True)
