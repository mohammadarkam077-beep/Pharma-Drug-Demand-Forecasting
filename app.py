"""
🏥 INDIA PHARMA FORECASTING DASHBOARD
======================================
Multi-page Streamlit dashboard showcasing:
- National demand forecasts with ensemble ML
- State-level priority analysis with real data
- Bayesian uncertainty quantification (PyMC)
- Interactive sensitivity & what-if analysis
- Executive summary with strategic insights

Author: Mohammad Arkam
Built with: Streamlit + Plotly + PyMC + scikit-learn
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ============================================================
# PAGE CONFIG (must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="India Pharma Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — PREMIUM DARK THEME WITH GRADIENTS
# ============================================================
st.markdown("""
<style>
    /* Root styles */
    .main {
        background-color: #0E1117;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1E3A5F 0%, #0F1E33 50%, #1a1a2e 100%);
        padding: 60px 40px;
        border-radius: 16px;
        margin-bottom: 40px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        text-align: center;
        border: 1px solid #334155;
    }
    
    .hero-title {
        font-size: 52px;
        font-weight: 800;
        color: #F1F5F9;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 20px;
        color: #94A3B8;
        margin-top: 12px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 30px;
        flex-wrap: wrap;
    }
    
    .hero-stat {
        text-align: center;
    }
    
    .hero-stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #10B981;
    }
    
    .hero-stat-label {
        font-size: 12px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Navigation cards */
    .nav-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 28px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        cursor: pointer;
        text-decoration: none;
        color: inherit;
        display: block;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .nav-card:hover {
        border-color: #10B981;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
        transform: translateY(-4px);
    }
    
    .nav-card-header {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .nav-card-icon {
        font-size: 36px;
        margin-right: 16px;
    }
    
    .nav-card-title {
        font-size: 22px;
        font-weight: 700;
        color: #F1F5F9;
        margin: 0;
    }
    
    .nav-card-description {
        color: #CBD5E1;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }
    
    .nav-card-meta {
        display: flex;
        gap: 16px;
        margin-top: 12px;
        font-size: 12px;
    }
    
    .nav-card-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        color: #F1F5F9;
        font-size: 28px;
        font-weight: 700;
        margin-top: 40px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid #334155;
    }
    
    /* USP boxes */
    .usp-box {
        background-color: #1E293B;
        padding: 24px;
        border-radius: 12px;
        border-left: 4px solid #10B981;
        margin-bottom: 16px;
        color: #CBD5E1;
    }
    
    .usp-box-title {
        color: #F1F5F9;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    /* Tech stack badges */
    .tech-badge {
        display: inline-block;
        background-color: #334155;
        color: #E2E8F0;
        padding: 8px 14px;
        border-radius: 6px;
        margin: 4px;
        font-size: 13px;
        font-weight: 600;
    }
    
    /* CTA button style */
    .cta-box {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-top: 30px;
    }
    
    .cta-text {
        color: white;
        font-size: 16px;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        padding: 30px 20px;
        font-size: 13px;
        border-top: 1px solid #334155;
        margin-top: 60px;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA (for hero stats)
# ============================================================
@st.cache_data
def load_master():
    path = Path("data/state_master.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_national():
    path = Path("outputs/forecast_output_advanced.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

df_master = load_master()
df_national = load_national()

# ============================================================
# COMPUTE HERO STATS
# ============================================================
total_market_cr = df_master['Revenue_Cr'].sum() if df_master is not None else 0
total_patients_M = df_master['Total_Diabetics'].sum() / 1e6 if df_master is not None else 0
num_states = len(df_master) if df_master is not None else 0

if df_national is not None:
    latest_fcst = df_national[df_national['Is_Forecast'] == 1].iloc[-1]
    first_yr = df_national.iloc[0]
    cagr = ((latest_fcst['Annual_Demand'] / first_yr['Annual_Demand']) ** (1/19) - 1) * 100
else:
    cagr = 0

# ============================================================
# HERO SECTION
# ============================================================
hero_html = """
<div class='hero-container'>
    <h1 class='hero-title'>💊 India Pharma Forecasting</h1>
    <p class='hero-subtitle'>Strategic Intelligence Platform for Diabetes Market Analysis</p>
    <div style='display:flex; justify-content:center; gap:40px; margin-top:30px; flex-wrap:wrap;'>
        <div style='text-align:center;'>
            <div style='font-size:28px; font-weight:700; color:#10B981;'>₹8,751 Cr</div>
            <div style='font-size:12px; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-top:4px;'>Market Size</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:28px; font-weight:700; color:#10B981;'>81.5M</div>
            <div style='font-size:12px; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-top:4px;'>Patients</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:28px; font-weight:700; color:#10B981;'>31+</div>
            <div style='font-size:12px; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-top:4px;'>States Modeled</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:28px; font-weight:700; color:#10B981;'>+96%</div>
            <div style='font-size:12px; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-top:4px;'>5Y Growth</div>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# ============================================================
# QUICK INSIGHTS BANNER
# ============================================================
st.markdown("""
<div style='background: linear-gradient(90deg, rgba(16,185,129,0.1) 0%, rgba(59,130,246,0.1) 100%);
            border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 40px;'>
    <p style='color: #CBD5E1; margin: 0; font-size: 15px;'>
        <span style='color: #10B981; font-weight: 700;'>🎯 Key Insight:</span>
        Untapped market opportunity of <b>₹24,466 Cr</b> (280% larger than current market) 
        with only 31% of diabetics currently diagnosed. Top 5 states represent 46% of revenue.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NAVIGATION CARDS
# ============================================================
st.markdown("<h2 class='section-header'>🗺️ Explore the Platform</h2>", unsafe_allow_html=True)

# Column layout for cards
col1, col2 = st.columns(2)

# Page 1: National Forecast
with col1:
    st.markdown("""
    <div class='nav-card'>
        <div class='nav-card-header'>
            <div class='nav-card-icon'>📈</div>
            <h3 class='nav-card-title'>National Forecast</h3>
        </div>
        <p class='nav-card-description'>
            20-year demand forecast using ensemble ML (ARIMA + Holt-Winters + XGBoost + Ridge + Lasso) 
            with Monte Carlo uncertainty quantification.
        </p>
        <div class='nav-card-meta'>
            <span class='nav-card-badge'>⏱️ 2010–2029</span>
            <span class='nav-card-badge'>📊 6 Models</span>
            <span class='nav-card-badge'>📈 Ensemble</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Page 2: State Explorer
with col2:
    st.markdown("""
    <div class='nav-card'>
        <div class='nav-card-header'>
            <div class='nav-card-icon'>🗺️</div>
            <h3 class='nav-card-title'>State Explorer</h3>
        </div>
        <p class='nav-card-description'>
            Geographic priority analysis with market tier ranking, healthcare access scoring, 
            and treatment gap identification across 31+ states.
        </p>
        <div class='nav-card-meta'>
            <span class='nav-card-badge'>🌍 31+ States</span>
            <span class='nav-card-badge'>📍 Geospatial</span>
            <span class='nav-card-badge'>🎯 Priority</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Page 3: Bayesian Uncertainty
with col1:
    st.markdown("""
    <div class='nav-card'>
        <div class='nav-card-header'>
            <div class='nav-card-icon'>🔬</div>
            <h3 class='nav-card-title'>Bayesian Uncertainty</h3>
        </div>
        <p class='nav-card-description'>
            Hierarchical Bayesian model (PyMC + nutpie) with state-level credible intervals, 
            regional effects, and posterior distributions. R-hat = 1.01 (excellent convergence).
        </p>
        <div class='nav-card-meta'>
            <span class='nav-card-badge'>🤖 Hierarchical</span>
            <span class='nav-card-badge'>✅ R-hat 1.01</span>
            <span class='nav-card-badge'>📊 95% CI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Page 4: Sensitivity Analysis
with col2:
    st.markdown("""
    <div class='nav-card'>
        <div class='nav-card-header'>
            <div class='nav-card-icon'>🎯</div>
            <h3 class='nav-card-title'>Sensitivity Analysis</h3>
        </div>
        <p class='nav-card-description'>
            Interactive what-if scenarios with live revenue recalculation. Tornado chart identifies 
            highest-leverage parameters. Compare pessimistic vs base vs optimistic outcomes.
        </p>
        <div class='nav-card-meta'>
            <span class='nav-card-badge'>🎚️ 6 Sliders</span>
            <span class='nav-card-badge'>💰 Live Calc</span>
            <span class='nav-card-badge'>🌪️ Tornado</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Page 5: Executive Summary
with col1:
    st.markdown("""
    <div class='nav-card'>
        <div class='nav-card-header'>
            <div class='nav-card-icon'>📋</div>
            <h3 class='nav-card-title'>Executive Summary</h3>
        </div>
        <p class='nav-card-description'>
            Stakeholder-ready 1-page overview with strategic insights, top 5 opportunities, 
            recommendations, and downloadable reports (TXT + CSV exports).
        </p>
        <div class='nav-card-meta'>
            <span class='nav-card-badge'>🏆 Top 5</span>
            <span class='nav-card-badge'>📊 Insights</span>
            <span class='nav-card-badge'>📥 Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='nav-card' style='opacity: 0.7; border-style: dashed;'>
        <div class='nav-card-header'>
            <div class='nav-card-icon'>🚀</div>
            <h3 class='nav-card-title'>Coming Soon</h3>
        </div>
        <p class='nav-card-description'>
            PDF report generation, email integration, real-time data refresh, 
            scenario comparison, and advanced analytics features.
        </p>
        <div class='nav-card-meta'>
            <span class='nav-card-badge'>🔜 Phase 2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# WHY THIS PLATFORM?
# ============================================================
st.markdown("<h2 class='section-header'>🌟 Why This Platform?</h2>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='usp-box'>
        <div class='usp-box-title'>📊 Real Data, Not Guesses</div>
        <p style='margin: 0; color: #94A3B8; font-size: 14px; line-height: 1.6;'>
            Integrated 6+ authoritative sources (IHME GBD, ICMR-INDIAB-17, NFHS-5, UN, World Bank) 
            for ground-truth validation.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='usp-box'>
        <div class='usp-box-title'>🤖 Rigorous Statistical Methods</div>
        <p style='margin: 0; color: #94A3B8; font-size: 14px; line-height: 1.6;'>
            Bayesian hierarchical modeling with proper uncertainty quantification—not just point estimates. 
            R-hat = 1.01 proves convergence.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='usp-box'>
        <div class='usp-box-title'>⚡ Interactive & Actionable</div>
        <p style='margin: 0; color: #94A3B8; font-size: 14px; line-height: 1.6;'>
            Real-time what-if scenarios, sensitivity analysis, and executive summaries ready to present 
            to leadership.
        </p>
    </div>
    """, unsafe_allow_html=True)   # ← FIXED

# ============================================================
# METHODOLOGY HIGHLIGHTS
# ============================================================
st.markdown("<h2 class='section-header'>🔬 Methodology & Tech Stack</h2>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class='usp-box' style='border-left-color: #3B82F6;'>
        <div class='usp-box-title'>📈 Forecasting Models</div>
        <ul style='color: #CBD5E1; font-size: 14px; margin: 8px 0; padding-left: 20px;'>
            <li>ARIMA (Auto-regressive)</li>
            <li>Holt-Winters (Exponential)</li>
            <li>XGBoost (Gradient Boosting)</li>
            <li>Ridge & Lasso Regression</li>
            <li>Ensemble voting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class='usp-box' style='border-left-color: #8B5CF6;'>
        <div class='usp-box-title'>🧮 Bayesian Inference</div>
        <ul style='color: #CBD5E1; font-size: 14px; margin: 8px 0; padding-left: 20px;'>
            <li>Hierarchical PyMC model</li>
            <li>nutpie sampler (Rust-based)</li>
            <li>Non-centered parameterization</li>
            <li>4 chains × 1,000 draws</li>
            <li>R-hat = 1.01 (converged)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class='usp-box' style='border-left-color: #10B981;'>
        <div class='usp-box-title'>🛠️ Tech Stack</div>
        <ul style='color: #CBD5E1; font-size: 14px; margin: 8px 0; padding-left: 20px;'>
            <li><strong>Frontend:</strong> Streamlit</li>
            <li><strong>Visualization:</strong> Plotly</li>
            <li><strong>ML:</strong> scikit-learn</li>
            <li><strong>Bayesian:</strong> PyMC + nutpie</li>
            <li><strong>Data:</strong> pandas, NumPy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DATA SOURCES
# ============================================================
st.markdown("<h2 class='section-header'>📚 Data Sources</h2>", unsafe_allow_html=True)

st.markdown("""
<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px;'>
    <div class='usp-box' style='border-left-color: #F59E0B;'>
        <div class='usp-box-title'>🌍 Global Health Data</div>
        <p style='color: #CBD5E1; font-size: 13px; margin: 0;'>
            <strong>IHME GBD 2023:</strong> Global Burden of Disease, 31 states, 2010–2023
        </p>
    </div>
    <div class='usp-box' style='border-left-color: #EC4899;'>
        <div class='usp-box-title'>🏥 National Survey</div>
        <p style='color: #CBD5E1; font-size: 13px; margin: 0;'>
            <strong>ICMR-INDIAB-17:</strong> India's largest diabetes prevalence study
        </p>
    </div>
    <div class='usp-box' style='border-left-color: #06B6D4;'>
        <div class='usp-box-title'>👨‍👩‍👧‍👦 Household Survey</div>
        <p style='color: #CBD5E1; font-size: 13px; margin: 0;'>
            <strong>NFHS-5:</strong> Health & nutrition data for all states
        </p>
    </div>
    <div class='usp-box' style='border-left-color: #10B981;'>
        <div class='usp-box-title'>🌐 Demographics</div>
        <p style='color: #CBD5E1; font-size: 13px; margin: 0;'>
            <strong>UN World Population Prospects:</strong> Age-stratified projections
        </p>
    </div>
    <div class='usp-box' style='border-left-color: #3B82F6;'>
        <div class='usp-box-title'>🏢 Infrastructure</div>
        <p style='color: #CBD5E1; font-size: 13px; margin: 0;'>
            <strong>data.gov.in:</strong> Hospital directory & healthcare facilities
        </p>
    </div>
    <div class='usp-box' style='border-left-color: #8B5CF6;'>
        <div class='usp-box-title'>💰 Economic Data</div>
        <p style='color: #CBD5E1; font-size: 13px; margin: 0;'>
            <strong>World Bank:</strong> Health expenditure & development indicators
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CTA SECTION
# ============================================================
st.markdown("""
<div class='cta-box'>
    <p class='cta-text'>
        👈 <strong>Use the sidebar to navigate through all pages</strong> or click any card above to explore!
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# AUTHOR & PROJECT INFO
# ============================================================
st.markdown("""
<div class='footer'>
    <p style='margin: 0; color: #94A3B8;'>
        <strong style='color: #E2E8F0;'>India Pharma Forecasting Platform</strong> 
        &nbsp;|&nbsp; Built by <strong>Mohammad Arkam</strong>
    </p>
    <p style='margin: 8px 0 0 0; color: #64748B; font-size: 12px;'>
        Powered by Streamlit • Plotly • PyMC • scikit-learn • pandas
        <br>
        <a href='https://github.com/mohammadarkam077-beep/Pharma-Drug-Demand-Forecasting' style='color: #3B82F6; text-decoration: none;'>GitHub</a>
        &nbsp;|&nbsp;
        <a href='https://www.linkedin.com/in/mohammad-arkam-917345197' style='color: #3B82F6; text-decoration: none;'>LinkedIn</a>
        &nbsp;|&nbsp;
        <a href='mailto:mohammadarkam077@email.com' style='color: #3B82F6; text-decoration: none;'>Contact</a>
    </p>
</div>
""", unsafe_allow_html=True)