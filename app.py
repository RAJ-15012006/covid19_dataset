import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_loader import load_and_preprocess_data, get_available_countries
from forecasting import fit_arima_forecast
from ml_models import (
    run_multi_model_forecast, train_wave_detector,
    predict_current_wave_risk, calculate_risk_score,
    calculate_vaccine_impact
)
from visualizations import (
    plot_3d_surface, plot_animated_choropleth, plot_dual_axis_line,
    plot_stacked_area, plot_vaccination_race, plot_correlation_heatmap,
    plot_forecast, COLORS, apply_custom_theme
)

# -----------------
# Page Setup & CSS
# -----------------
st.set_page_config(layout="wide", page_title="COVID-19 Predictive Intelligence", page_icon="🦠")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Rajdhani:wght@500;700&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
        color: #e0e0e0;
        background-color: #0d1117;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 10% 10%, #0d1b2a 0%, #11001c 50%, #0a0a0a 100%) !important;
    }
    /* floating virus decorations */
    .stApp::before {
        content: "\1F9A0";
        position: fixed; font-size: 140px; opacity: 0.05;
        top: 4%; left: 2%;
        animation: floatV 9s ease-in-out infinite;
        z-index: -1; pointer-events: none;
    }
    .stApp::after {
        content: "\1F9A0";
        position: fixed; font-size: 200px; opacity: 0.04;
        bottom: 6%; right: 3%;
        animation: floatV 13s ease-in-out infinite reverse;
        z-index: -1; pointer-events: none;
    }
    @keyframes floatV {
        0%   { transform: translateY(0px) rotate(0deg); }
        50%  { transform: translateY(-28px) rotate(18deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(120deg, #0d47a1, #4a148c, #880e4f);
        border-radius: 20px;
        padding: 40px 50px;
        margin-bottom: 30px;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 60px rgba(0,245,255,0.15);
        border: 1px solid rgba(0,245,255,0.15);
    }
    .hero-banner::before {
        content: "\1F9A0  \1F9EC  \1F489  \1F52C  \1F9A0  \1F9EC  \1F489  \1F52C";
        position: absolute; font-size: 26px; opacity: 0.1;
        top: 10px; left: 0; letter-spacing: 20px;
        white-space: nowrap;
        animation: scrollTicker 22s linear infinite;
    }
    @keyframes scrollTicker {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .hero-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3rem; font-weight: 700; margin: 0;
        color: #00f5ff !important;
        text-shadow: 0 0 20px rgba(0,245,255,0.5);
    }
    .hero-sub { font-size: 1rem; opacity: 0.85; margin-top: 8px; color: #ccc; }
    .badge {
        display: inline-block;
        background: rgba(0,245,255,0.1);
        border: 1px solid rgba(0,245,255,0.35);
        color: #00f5ff;
        border-radius: 20px;
        padding: 4px 14px; font-size: 0.78rem; margin: 4px 4px 0 0;
    }

    /* ── Metric Cards ── */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 16px !important;
        border-left: 5px solid #00f5ff !important;
        box-shadow: 0 4px 24px rgba(0,245,255,0.08) !important;
        padding: 18px !important;
        transition: transform 0.25s ease;
        backdrop-filter: blur(6px);
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-4px); }
    div[data-testid="stMetricValue"] > div {
        color: #00f5ff !important;
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem; font-weight: 700;
        text-shadow: 0 0 10px rgba(0,245,255,0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #aaa !important;
        font-size: 0.82rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid rgba(0,245,255,0.12) !important;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: #161b22;
        padding: 10px 14px; border-radius: 14px;
        border: 1px solid rgba(0,245,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px; background: #0d1117;
        border-radius: 10px; color: #aaa;
        font-weight: 600; border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,245,255,0.12) !important;
        color: #00f5ff !important;
        border-color: #00f5ff !important;
    }

    /* ── Section Cards ── */
    .section-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(0,245,255,0.1);
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
    }

    /* ── Insight Box ── */
    .insight-box {
        background: rgba(0,245,255,0.05);
        border-left: 4px solid #00f5ff;
        border-radius: 10px; padding: 14px 18px;
        margin: 10px 0; font-size: 0.9rem; color: #ccc;
    }

    /* ── Skill Tags ── */
    .skill-tag {
        display: inline-block;
        background: rgba(0,245,255,0.1);
        color: #00f5ff;
        border: 1px solid rgba(0,245,255,0.3);
        border-radius: 20px; padding: 5px 14px;
        font-size: 0.78rem; margin: 3px; font-weight: 600;
    }

    /* ── Tech stack bar ── */
    .tech-bar {
        background: #161b22;
        border: 1px solid rgba(0,245,255,0.1);
        border-radius: 14px; padding: 16px 24px; margin-bottom: 20px;
    }

    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700; color: #ffffff !important;
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #00f5ff; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# -----------------
# Data Loading
# -----------------
with st.status("🔄 Loading COVID-19 dataset... (First run may take a few minutes to download ~400MB)", expanded=True) as status:
    df = load_and_preprocess_data()
    if not df.empty:
        status.update(label="✅ Data loaded successfully!", state="complete", expanded=False)

if df.empty:
    st.error("Data failed to load. Please verify your connection or local 'owid-covid-data.csv' file.")
    st.stop()

# -----------------
# Sidebar Controls
# -----------------
st.sidebar.header("🎯 Predictive Intel")

countries_list = get_available_countries(df)
default_countries = [c for c in ["United States", "India", "Brazil", "United Kingdom", "Germany"] if c in countries_list]

selected_countries = st.sidebar.multiselect(
    "Select Countries (Comparison):",
    options=countries_list,
    default=default_countries[:5],
    max_selections=5
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.slider(
    "Date Range (2020-2023):",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

metric_map = {"Cases": "new_cases", "Deaths": "new_deaths", "Vaccinations": "total_vaccinations"}
selected_metric_label = st.sidebar.selectbox("Core Metric:", options=list(metric_map.keys()))
selected_metric = metric_map[selected_metric_label]

st.sidebar.markdown("---")
st.sidebar.caption("System Status: Online 🛰️")
st.sidebar.caption("Data: OWID (May 2023 Cutoff)")

# -----------------
# Hero Banner
# -----------------
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🦠 COVID-19 Predictive Intelligence Platform</p>
    <p class="hero-sub">End-to-end ML pipeline • Real-time risk scoring • Causal inference • Multi-model forecasting</p>
    <div style="margin-top:14px">
        <span class="badge">🤖 XGBoost + ARIMA</span>
        <span class="badge">🌍 200+ Countries</span>
        <span class="badge">📈 Time-Series Forecasting</span>
        <span class="badge">💉 Vaccine Impact Analysis</span>
        <span class="badge">⚠️ Wave Detection RF</span>
        <span class="badge">🔬 Causal Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Tech Stack Banner
st.markdown("""
<div class="tech-bar">
    <span style="font-size:0.8rem; font-weight:700; color:#555; text-transform:uppercase; letter-spacing:1px;">Tech Stack &amp; Skills Demonstrated</span><br/>
    <span class="skill-tag">Python</span>
    <span class="skill-tag">Streamlit</span>
    <span class="skill-tag">XGBoost</span>
    <span class="skill-tag">ARIMA / statsmodels</span>
    <span class="skill-tag">Random Forest</span>
    <span class="skill-tag">Scikit-learn</span>
    <span class="skill-tag">Plotly</span>
    <span class="skill-tag">Pandas / NumPy</span>
    <span class="skill-tag">Causal Inference</span>
    <span class="skill-tag">Feature Engineering</span>
    <span class="skill-tag">Data Visualization</span>
    <span class="skill-tag">ML Pipeline Design</span>
</div>
""", unsafe_allow_html=True)

# -----------------
# App Layout - Tabs
# -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌎 Global Analytics",
    "📈 Forecast Leaderboard",
    "🚨 Wave & Risk Engine",
    "💉 Vaccine Impact",
    "📊 Insights Summary"
])

# Shared Filter
mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
df_filtered = df[mask].copy()
primary_country = selected_countries[0] if selected_countries else "United States"
if primary_country not in df["location"].values:
    primary_country = "United States"
df_primary = df_filtered[df_filtered["location"] == primary_country].sort_values("date")

# -----------------
# TAB 1: GLOBAL ANALYTICS
# -----------------
with tab1:
    st.subheader(f"🌎 Global Trends & {primary_country} Profile")

    # KPIs
    if not df_primary.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        latest = df_primary.iloc[-1]
        c1.metric("🦠 Total Cases", f"{int(latest['total_cases']):,}")
        c2.metric("❤️ Total Deaths", f"{int(latest['total_deaths']):,}")
        c3.metric("⚠️ CFR %", f"{(latest['total_deaths']/latest['total_cases']*100):.2f}%")
        vax_pct = (latest['people_fully_vaccinated']/latest['population']*100) if latest['population'] > 0 else 0
        c4.metric("💉 Vax Coverage", f"{vax_pct:.1f}%")
        c5.metric("📅 Data Through", str(latest['date'])[:10])

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_animated_choropleth(df_filtered, selected_metric), use_container_width=True)
        if not df_primary.empty:
            st.plotly_chart(plot_dual_axis_line(df_primary, selected_metric), use_container_width=True)
    with col2:
        countries_for_3d = selected_countries if selected_countries else [primary_country]
        st.plotly_chart(plot_3d_surface(df_filtered, countries_for_3d, selected_metric), use_container_width=True)
        if not df_primary.empty:
            st.plotly_chart(plot_correlation_heatmap(df_primary), use_container_width=True)

# -----------------
# TAB 2: FORECAST LEADERBOARD
# -----------------
with tab2:
    st.subheader("🥇 Model Comparison: ARIMA vs XGBoost")
    st.markdown("""
    <div class="insight-box">
    💡 <b>What this shows:</b> Classical statistical forecasting (ARIMA) vs modern gradient boosting (XGBoost) — evaluated on MAE, RMSE, and MAPE on a held-out 30-day validation window.
    </div>""", unsafe_allow_html=True)
    
    if primary_country:
        with st.spinner(f"Training models for {primary_country}..."):
            full_df_primary = df[df["location"] == primary_country].sort_values("date")
            arima_res, xgb_res, leaderboard_df, err = run_multi_model_forecast(df, primary_country, selected_metric)
            
            if err:
                st.warning(err)
            else:
                st.table(leaderboard_df)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xgb_res["date"], y=xgb_res["actual"], name="Actual", line=dict(color=COLORS["cyan"])))
                fig.add_trace(go.Scatter(x=xgb_res["date"], y=xgb_res["forecast"], name="XGBoost (ML)", line=dict(color=COLORS["gold"], dash="dash")))
                if arima_res is not None:
                    fig.add_trace(go.Scatter(x=arima_res["date"], y=arima_res["forecast"], name="ARIMA (Stats)", line=dict(color=COLORS["coral"], dash="dot")))
                
                fig.update_layout(title=f"Head-to-Head: {selected_metric_label} Prediction")
                st.plotly_chart(apply_custom_theme(fig), use_container_width=True)

# -----------------
# TAB 3: WAVE & RISK ENGINE
# -----------------
with tab3:
    st.subheader("🚨 Early Warning System & Risk Mapping")
    st.markdown("""
    <div class="insight-box">
    💡 <b>How it works:</b> A Random Forest classifier trained on 7-day case growth velocity, vaccination rate, stringency index, and positivity rate to predict the probability of a new wave in the next 30 days.
    </div>""", unsafe_allow_html=True)
    
    col_w1, col_w2 = st.columns([1, 2])
    
    with col_w1:
        st.markdown("#### Wave Detector (RF)")
        rf_model = train_wave_detector(df)
        prob, label = predict_current_wave_risk(df_primary, rf_model)
        st.metric("30-Day Wave Prob", f"{prob:.1f}%", label)
        
        st.markdown("#### Risk Breakdown")
        total_risk, breakdown = calculate_risk_score(df_primary)
        st.progress(total_risk / 100)
        for factor, val in breakdown.items():
            st.write(f"{factor}: {val}")

    with col_w2:
        st.markdown("#### Country Risk Matrix")
        # Lightweight risk map calculation
        risk_data = []
        for c in selected_countries:
            score, _ = calculate_risk_score(df[df["location"] == c])
            risk_data.append({"Country": c, "Risk Score": score})
        
        fig_risk = px.bar(pd.DataFrame(risk_data), x="Country", y="Risk Score", color="Risk Score", color_continuous_scale="Reds")
        st.plotly_chart(apply_custom_theme(fig_risk), use_container_width=True)

# -----------------
# TAB 4: VACCINE IMPACT
# -----------------
with tab4:
    st.subheader("💉 Causal Inference: Vaccine Impact Analysis")
    st.markdown("""
    <div class="insight-box">
    💡 <b>Methodology:</b> A linear regression model is trained on 2020 (pre-vaccine) data to learn the case-to-death relationship. This model is then projected onto 2021+ data to estimate <i>counterfactual deaths</i> — what would have happened without vaccines.
    </div>""", unsafe_allow_html=True)
    
    impact_df, lives_saved = calculate_vaccine_impact(df_primary)
    
    if impact_df is not None:
        st.metric("Estimated Lives Saved", f"{int(lives_saved):,}", delta="Counterfactual Analysis", delta_color="normal")
        
        fig_impact = go.Figure()
        fig_impact.add_trace(go.Scatter(x=impact_df["date"], y=impact_df["counterfactual_deaths"], name="Predicted (No Vax)", fill='tozeroy', line=dict(color=COLORS["coral"])))
        fig_impact.add_trace(go.Scatter(x=impact_df["date"], y=impact_df["new_deaths"], name="Actual (Post-Vax)", fill='tozeroy', line=dict(color=COLORS["cyan"])))
        
        fig_impact.update_layout(title=f"Averting Mortality: {primary_country}")
        st.plotly_chart(apply_custom_theme(fig_impact), use_container_width=True)
    else:
        st.warning("Insufficient historical (2020) or vaccine (2021+) data for this analysis.")

# -----------------
# TAB 5: INSIGHTS SUMMARY (Recruiter-Friendly)
# -----------------
with tab5:
    st.subheader("📊 Project Insights & Technical Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="section-card">
            <h3>🎯 What This Project Demonstrates</h3>
            <div class="insight-box">🔬 <b>End-to-End ML Pipeline</b> — Data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment in a single Streamlit app.</div>
            <div class="insight-box">🤖 <b>Multi-Model Forecasting</b> — ARIMA (statistical) vs XGBoost (ML) with a live leaderboard comparing MAE, RMSE, and MAPE.</div>
            <div class="insight-box">⚠️ <b>Wave Detection System</b> — Random Forest classifier with engineered features (growth velocity, vaccination rate, stringency) for early warning.</div>
            <div class="insight-box">💉 <b>Causal Inference</b> — Counterfactual analysis estimating lives saved by vaccines using a pre/post intervention regression model.</div>
            <div class="insight-box">🌎 <b>Global Visualizations</b> — Animated choropleth maps, 3D surface plots, dual-axis charts, and correlation heatmaps across 200+ countries.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="section-card">
            <h3>🛠️ Architecture Overview</h3>
            <div class="insight-box">📂 <b>data_loader.py</b> — Loads OWID dataset, handles missing values, engineers derived features (active cases, CFR, rolling averages).</div>
            <div class="insight-box">📈 <b>forecasting.py</b> — ARIMA model with auto-order selection, confidence intervals, and validation metrics.</div>
            <div class="insight-box">🤖 <b>ml_models.py</b> — XGBoost with lag features, Random Forest wave classifier, risk scorer, and vaccine impact estimator.</div>
            <div class="insight-box">🎨 <b>visualizations.py</b> — Plotly-based interactive charts with consistent theming and reusable components.</div>
            <div class="insight-box">🚀 <b>app.py</b> — Streamlit orchestration layer with sidebar controls, tab navigation, and responsive layout.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Key Numbers
    st.markdown("### 📊 Key Numbers at a Glance")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🌍 Countries Covered", "200+")
    k2.metric("🤖 ML Models", "3")
    k3.metric("📅 Years of Data", "2020–2023")
    k4.metric("📊 Chart Types", "8+")
    k5.metric("⚡ Features Engineered", "15+")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d47a1, #4a148c); border-radius:14px; padding:24px; color:white; text-align:center; border:1px solid rgba(0,245,255,0.2);">
        <h3 style="color:#00f5ff !important; margin:0 0 8px 0;">📞 Open to Data Science &amp; ML Engineering Roles</h3>
        <p style="opacity:0.85; margin:0; color:#ccc;">This project showcases skills in machine learning, time-series analysis, causal inference, and full-stack data app development.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #888; font-size:0.82rem;'>
    🦠 COVID-19 Predictive Intelligence Platform &nbsp;|&nbsp; Built with Python, Streamlit & Plotly &nbsp;|&nbsp; 2025
</p>
""", unsafe_allow_html=True)
