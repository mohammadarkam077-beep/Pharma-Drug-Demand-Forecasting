"""
🎯 SENSITIVITY & WHAT-IF ANALYSIS
==================================
Interactive scenario modeling with live revenue recalculation.
Uses real Bayesian state-level prevalence + state_master.csv (Adult_Pop, Population_2024).

Author: Mohammad Arkam
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sensitivity & What-If",
    page_icon="🎯",
    layout="wide"
)

# ============================================================
# CUSTOM CSS — MATCHES app.py / Bayesian.py
# ============================================================
st.markdown("""
<style>
    .section-header {
        color: #F1F5F9;
        font-size: 24px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #334155;
    }
    .info-box {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 10px 0;
        color: #CBD5E1;
    }
    .insight-box {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin: 10px 0;
        color: #CBD5E1;
    }
    .warning-box {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
        margin: 10px 0;
        color: #CBD5E1;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #0F1E33 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #F59E0B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_bayesian_intervals():
    path = Path("outputs/bayesian_state_intervals.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_state_master():
    path = Path("data/state_master.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)

df_bayes = load_bayesian_intervals()
df_master = load_state_master()

if df_bayes is None:
    st.error("⚠️ Could not find `outputs/bayesian_state_intervals.csv`. "
             "Run `python bayesian_forecasting.py` first.")
    st.stop()

if df_master is None:
    st.error("⚠️ Could not find `data/state_master.csv`. Run `python process_state_data.py` first.")
    st.stop()

# Merge Bayesian intervals + state master (real population & baseline data)
master_keep = ['State', 'Region', 'Tier', 'Population_2024', 'Adult_Pop',
               'Total_Diabetics', 'Diagnosis_Rate_Adjusted', 'Treatment_Rate_Adjusted',
               'Diagnosed_Patients', 'Treated_Patients', 'Annual_Revenue',
               'Revenue_Cr', 'Opportunity_Revenue_Cr', 'Priority_Score', 'Market_Tier']
master_keep = [c for c in master_keep if c in df_master.columns]
df = df_bayes.merge(df_master[master_keep], on='State', how='left',
                     suffixes=('', '_master'))

# Pre-existing baseline revenue from state_master (for validation)
ACTUAL_BASELINE_REVENUE_CR = df['Revenue_Cr'].sum() if 'Revenue_Cr' in df.columns else None

# Sidebar info
with st.sidebar:
    st.markdown("### 📊 Data Sources")
    st.caption(f"✅ Bayesian intervals: **{len(df_bayes)} states**")
    st.caption(f"✅ State master: **{len(df_master)} states**")
    st.caption(f"✅ Merged: **{len(df)} states**")
    if ACTUAL_BASELINE_REVENUE_CR:
        st.caption(f"💰 Actual baseline: **₹{ACTUAL_BASELINE_REVENUE_CR:,.0f} Cr**")

# ============================================================
# HEADER
# ============================================================
st.title("🎯 Sensitivity & What-If Analysis")
st.markdown("""
**Interactive scenario modeling** — adjust market & product parameters and see live revenue impact across all states.
""")
st.markdown("---")

# ============================================================
# OVERVIEW BOX
# ============================================================
st.markdown(f"""
<div class='info-box'>
    <h4 style='color: #FBBF24; margin-top: 0;'>🎚️ How This Works</h4>
    <ul style='margin-bottom: 0;'>
        <li><b>Inputs:</b> Bayesian state prevalence + 6 adjustable market/product parameters</li>
        <li><b>Funnel:</b> Adult Pop → Diabetic → Diagnosed → Accessible → Addressable → Treated → Revenue</li>
        <li><b>Live Recalculation:</b> Every slider movement triggers full state-level recompute across {len(df)} states</li>
        <li><b>Tornado Chart:</b> Identifies which parameter has the highest revenue leverage</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 1: PARAMETER SLIDERS
# ============================================================
st.markdown("<div class='section-header'>🎚️ Adjust Parameters</div>", unsafe_allow_html=True)
st.caption("Move sliders to see live revenue impact")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 Market Parameters")
    prevalence_adj = st.slider("Diabetes Prevalence Adjustment (%)",
        -30, 30, 0, 1,
        help="± shift from Bayesian point estimate (within credible interval)")
    market_access = st.slider("Market Access (%)", 10, 95, 55, 5,
        help="% of diagnosed patients who can access the drug")
    diagnosis_rate = st.slider("Diagnosis Rate (%)", 20, 90, 50, 5,
        help="% of diabetic population that is actually diagnosed")

with col2:
    st.markdown("#### 💊 Product Parameters")
    price_per_patient = st.slider("Annual Price per Patient (₹)",
        5000, 50000, 18000, 1000,
        help="Annual treatment cost per patient")
    adherence = st.slider("Patient Adherence (%)", 30, 95, 65, 5,
        help="% of patients who stay on therapy for the full year")
    market_share = st.slider("Target Market Share (%)", 1, 40, 8, 1,
        help="Your product's share of the addressable market")

# ============================================================
# CORE CALCULATION (uses REAL Adult_Pop)
# ============================================================
def calculate_revenue(df_in, prev_adj, access, dx_rate, price, adh, share):
    """
    Revenue funnel calculation per state.
    Mean_Prevalence is decimal (e.g. Kerala = 0.28683).
    Adult_Pop is the real adult population from state_master.csv.
    """
    rows = []
    for _, row in df_in.iterrows():
        adult_pop = row['Adult_Pop']
        if pd.isna(adult_pop):
            continue
        
        adj_prev = row['Mean_Prevalence'] * (1 + prev_adj / 100)
        adj_prev = max(0.01, min(0.60, adj_prev))
        
        diabetic    = adult_pop * adj_prev
        diagnosed   = diabetic * (dx_rate / 100)
        accessible  = diagnosed * (access / 100)
        addressable = accessible * (share / 100)
        treated     = addressable * (adh / 100)
        revenue     = treated * price
        
        rows.append({
            'State': row['State'],
            'Adult_Pop': adult_pop,
            'Diabetic': diabetic,
            'Treated_Patients': treated,
            'Revenue_Cr': revenue / 1e7,
        })
    return pd.DataFrame(rows)

# Compute current scenario + baseline
scenario_df = calculate_revenue(df, prevalence_adj, market_access, diagnosis_rate,
                                 price_per_patient, adherence, market_share)
baseline_df = calculate_revenue(df, 0, 55, 50, 18000, 65, 8)

total_revenue   = scenario_df['Revenue_Cr'].sum()
baseline_rev    = baseline_df['Revenue_Cr'].sum()
revenue_delta   = total_revenue - baseline_rev
total_patients  = scenario_df['Treated_Patients'].sum()
pct_change      = (revenue_delta / baseline_rev * 100) if baseline_rev > 0 else 0
revenue_per_pat = (total_revenue * 1e7 / total_patients) if total_patients > 0 else 0

# ============================================================
# SECTION 2: LIVE KPI CARDS
# ============================================================
st.markdown("<div class='section-header'>💰 Live Revenue Projection</div>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div style='color:#94A3B8;font-size:14px;text-transform:uppercase;'>Total Revenue</div>
        <div style='color:#F1F5F9;font-size:28px;font-weight:700;'>₹{total_revenue:,.0f} Cr</div>
        <div style='color:{"#10B981" if revenue_delta >= 0 else "#EF4444"};font-size:13px;'>
            {revenue_delta:+,.0f} Cr vs baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div style='color:#94A3B8;font-size:14px;text-transform:uppercase;'>Treated Patients</div>
        <div style='color:#F1F5F9;font-size:28px;font-weight:700;'>{total_patients/1e6:.2f}M</div>
        <div style='color:#94A3B8;font-size:13px;'>Annual</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div style='color:#94A3B8;font-size:14px;text-transform:uppercase;'>Revenue / Patient</div>
        <div style='color:#F1F5F9;font-size:28px;font-weight:700;'>₹{revenue_per_pat:,.0f}</div>
        <div style='color:#94A3B8;font-size:13px;'>Annual</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    color = "#10B981" if pct_change > 0 else "#EF4444" if pct_change < 0 else "#94A3B8"
    label = "Better" if pct_change > 0 else "Worse" if pct_change < 0 else "Same as Baseline"
    st.markdown(f"""
    <div class='kpi-card'>
        <div style='color:#94A3B8;font-size:14px;text-transform:uppercase;'>vs Baseline</div>
        <div style='color:#F1F5F9;font-size:28px;font-weight:700;'>{pct_change:+.1f}%</div>
        <div style='color:{color};font-size:13px;'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

# Show comparison to actual market revenue if available
if ACTUAL_BASELINE_REVENUE_CR:
    market_capture_pct = (total_revenue / ACTUAL_BASELINE_REVENUE_CR) * 100
    
    # Determine status emoji based on capture %
    if market_capture_pct < 15:
        status_emoji = "🌱"
        status_text = "Niche player"
        status_color = "#94A3B8"
    elif market_capture_pct < 35:
        status_emoji = "📈"
        status_text = "Growing share"
        status_color = "#F59E0B"
    elif market_capture_pct < 60:
        status_emoji = "🚀"
        status_text = "Strong position"
        status_color = "#10B981"
    else:
        status_emoji = "👑"
        status_text = "Market leader"
        status_color = "#8B5CF6"
    
    st.markdown(f"""
    <div class='info-box' style='margin-top: 15px;'>
        📌 <b>Total Indian Diabetes Market:</b> ₹{ACTUAL_BASELINE_REVENUE_CR:,.0f} Cr
        &nbsp;|&nbsp; Your scenario captures
        <b style='color:{status_color};'>{market_capture_pct:.1f}%</b>
        of the total market &nbsp;{status_emoji} <i>{status_text}</i>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SECTION 3: STATE REVENUE BAR CHART
# ============================================================
st.markdown("<div class='section-header'>🗺️ Revenue by State (Current Scenario)</div>", unsafe_allow_html=True)

scen_sorted = scenario_df.sort_values('Revenue_Cr', ascending=True)
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    y=scen_sorted['State'],
    x=scen_sorted['Revenue_Cr'],
    orientation='h',
    marker=dict(
        color=scen_sorted['Revenue_Cr'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='₹ Cr', x=1.0)
    ),
    text=[f"₹{v:,.1f} Cr" for v in scen_sorted['Revenue_Cr']],
    textposition='outside',
))
fig_bar.update_layout(
    template='plotly_dark',
    height=max(500, 25 * len(scen_sorted)),
    xaxis_title='Revenue (₹ Crores)',
    yaxis_title='',
    showlegend=False,
    margin=dict(l=0, r=80, t=30, b=0)
)
st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# SECTION 4: TORNADO CHART
# ============================================================
st.markdown("<div class='section-header'>🌪️ Tornado Chart — Parameter Sensitivity</div>", unsafe_allow_html=True)
st.caption("Each parameter is varied across its **realistic business uncertainty range** (industry benchmarks). "
           "Why not flat ±20%? Because revenue is multiplicative — flat ±20% on any factor produces identical swings, giving zero insight.")

def tornado_analysis(df_in, base_params):
    base_revenue = calculate_revenue(df_in, **base_params)['Revenue_Cr'].sum()
    
    param_ranges = {
        'prev_adj': {'label': 'Prevalence (Bayesian CI)', 'low': -25, 'high': 25},
        'access':   {'label': 'Market Access',            'low': 30,  'high': 85},
        'dx_rate':  {'label': 'Diagnosis Rate',           'low': 35,  'high': 75},
        'price':    {'label': 'Price per Patient',        'low': 10000, 'high': 30000},
        'adh':      {'label': 'Adherence',                'low': 40,  'high': 85},
        'share':    {'label': 'Market Share',             'low': 3,   'high': 20},
    }
    
    rows = []
    for key, info in param_ranges.items():
        p_low  = {**base_params, key: info['low']}
        p_high = {**base_params, key: info['high']}
        rev_low  = calculate_revenue(df_in, **p_low)['Revenue_Cr'].sum()
        rev_high = calculate_revenue(df_in, **p_high)['Revenue_Cr'].sum()
        rows.append({
            'param': info['label'],
            'low_value': info['low'], 'high_value': info['high'],
            'low_impact': rev_low - base_revenue,
            'high_impact': rev_high - base_revenue,
            'range': abs(rev_high - rev_low)
        })
    return pd.DataFrame(rows).sort_values('range', ascending=True)

base_params = {
    'prev_adj': prevalence_adj, 'access': market_access, 'dx_rate': diagnosis_rate,
    'price': price_per_patient, 'adh': adherence, 'share': market_share
}
tornado_df = tornado_analysis(df, base_params)

fig_tornado = go.Figure()
fig_tornado.add_trace(go.Bar(
    y=tornado_df['param'], x=tornado_df['low_impact'],
    orientation='h', name='Downside', marker_color='#EF4444',
    text=[f"₹{v:+,.0f} Cr" for v in tornado_df['low_impact']],
    textposition='outside'
))
fig_tornado.add_trace(go.Bar(
    y=tornado_df['param'], x=tornado_df['high_impact'],
    orientation='h', name='Upside', marker_color='#10B981',
    text=[f"₹{v:+,.0f} Cr" for v in tornado_df['high_impact']],
    textposition='outside'
))
fig_tornado.update_layout(
    template='plotly_dark', height=500, barmode='overlay',
    xaxis_title='Revenue Impact vs Current Scenario (₹ Crores)',
    legend=dict(orientation='h', y=-0.15),
    margin=dict(l=0, r=0, t=30, b=0)
)
fig_tornado.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)
st.plotly_chart(fig_tornado, use_container_width=True)

with st.expander("📊 See parameter ranges used"):
    rng = tornado_df[['param', 'low_value', 'high_value', 'range']].copy()
    rng.columns = ['Parameter', 'Low Value', 'High Value', 'Revenue Swing (₹ Cr)']
    rng['Revenue Swing (₹ Cr)'] = rng['Revenue Swing (₹ Cr)'].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(rng, hide_index=True, use_container_width=True)

top = tornado_df.iloc[-1]
st.markdown(f"""
<div class='insight-box'>
💡 <b>Key Insight:</b> <span style='color:#FBBF24;'>{top['param']}</span> is the highest-leverage parameter,
with a revenue swing of <b>₹{top['range']:,.0f} Cr</b> across its realistic range
({top['low_value']} → {top['high_value']}). Prioritize commercial investment here for maximum ROI.
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 5: SCENARIO COMPARISON
# ============================================================
st.markdown("<div class='section-header'>🎭 Scenario Comparison</div>", unsafe_allow_html=True)

is_custom = (prevalence_adj != 0 or market_access != 55 or diagnosis_rate != 50
             or price_per_patient != 18000 or adherence != 65 or market_share != 8)

if not is_custom:
    st.markdown("""
    <div class='warning-box'>
    ⚠️ Your sliders are at <b>default (Base Case)</b> values. Adjust them above to see "Your Scenario" differ.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='insight-box'>
    ✅ Custom scenario active — see how it compares to standard cases below.
    </div>
    """, unsafe_allow_html=True)

scenarios = {
    "😟 Pessimistic":    {'prev_adj': -15, 'access': 35, 'dx_rate': 35, 'price': 14000, 'adh': 45, 'share': 5},
    "😐 Base Case":      {'prev_adj':   0, 'access': 55, 'dx_rate': 50, 'price': 18000, 'adh': 65, 'share': 8},
    "😎 Optimistic":     {'prev_adj':  15, 'access': 75, 'dx_rate': 70, 'price': 22000, 'adh': 80, 'share': 12},
    "🎯 Your Scenario":  base_params,
}

scen_results = []
for name, params in scenarios.items():
    calc = calculate_revenue(df, **params)
    scen_results.append({
        'Scenario': name,
        'Revenue (₹ Cr)': calc['Revenue_Cr'].sum(),
        'Patients (M)':   calc['Treated_Patients'].sum() / 1e6,
    })
scen_df = pd.DataFrame(scen_results)

c1, c2 = st.columns([2, 1])
with c1:
    colors      = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6']
    line_widths = [0, 0, 0, 4 if is_custom else 0]
    line_colors = ['rgba(0,0,0,0)'] * 3 + ['#FFFFFF' if is_custom else 'rgba(0,0,0,0)']
    
    fig_scen = go.Figure()
    fig_scen.add_trace(go.Bar(
        x=scen_df['Scenario'], y=scen_df['Revenue (₹ Cr)'],
        marker=dict(color=colors, line=dict(width=line_widths, color=line_colors)),
        text=[f"₹{v:,.0f} Cr" for v in scen_df['Revenue (₹ Cr)']],
        textposition='outside'
    ))
    fig_scen.update_layout(
        template='plotly_dark', height=420,
        yaxis_title='Revenue (₹ Crores)', showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_scen, use_container_width=True)

with c2:
    st.markdown("#### 📋 Summary")
    disp = scen_df.copy()
    disp['Revenue (₹ Cr)'] = disp['Revenue (₹ Cr)'].apply(lambda x: f"₹{x:,.0f}")
    disp['Patients (M)']  = disp['Patients (M)'].apply(lambda x: f"{x:.2f}M")
    st.dataframe(disp, hide_index=True, use_container_width=True)
    
    your_rev = scen_df.loc[scen_df['Scenario'] == "🎯 Your Scenario", 'Revenue (₹ Cr)'].iloc[0]
    pess_rev = scen_df.loc[scen_df['Scenario'] == "😟 Pessimistic",   'Revenue (₹ Cr)'].iloc[0]
    opt_rev  = scen_df.loc[scen_df['Scenario'] == "😎 Optimistic",    'Revenue (₹ Cr)'].iloc[0]
    if opt_rev > pess_rev:
        pos = max(0, min(100, ((your_rev - pess_rev) / (opt_rev - pess_rev)) * 100))
        st.metric("Your scenario position", f"{pos:.0f}%", delta="Pessimistic → Optimistic")

# ============================================================
# SECTION 6: STATE-LEVEL DEEP DIVE
# ============================================================
st.markdown("<div class='section-header'>🔍 State-Level Deep Dive</div>", unsafe_allow_html=True)

selected_state = st.selectbox("Pick a state to analyze",
                               options=sorted(df['State'].tolist()), index=0)

state_row     = df[df['State'] == selected_state].iloc[0]
state_scen    = scenario_df[scenario_df['State'] == selected_state].iloc[0]
state_base    = baseline_df[baseline_df['State'] == selected_state].iloc[0]

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Adult Population", f"{state_row['Adult_Pop']/1e6:.1f}M")
with s2:
    st.metric("Diabetes Prevalence",
              f"{state_row['Mean_Prevalence']*100:.2f}%",
              delta=f"95% CI: [{state_row['Lower_95_CI']*100:.1f}, {state_row['Upper_95_CI']*100:.1f}]",
              delta_color="off")
with s3:
    delta_pat = ((state_scen['Treated_Patients'] / state_base['Treated_Patients'] - 1) * 100
                 if state_base['Treated_Patients'] > 0 else 0)
    st.metric("Treated Patients", f"{state_scen['Treated_Patients']:,.0f}",
              delta=f"{delta_pat:+.1f}% vs baseline")
with s4:
    st.metric("State Revenue", f"₹{state_scen['Revenue_Cr']:,.1f} Cr",
              delta=f"{(state_scen['Revenue_Cr'] - state_base['Revenue_Cr']):+,.1f} Cr")

# Show state's market tier and rank from master
if 'Market_Tier' in state_row.index and 'Priority_Score' in state_row.index:
    st.markdown(f"""
    <div class='info-box'>
    🏆 <b>{selected_state}</b> — Market Tier: <b>{state_row.get('Market_Tier', 'N/A')}</b>
    &nbsp;|&nbsp; Priority Score: <b>{state_row.get('Priority_Score', 0):.2f}</b>
    &nbsp;|&nbsp; Region: <b>{state_row.get('Region', 'N/A')}</b>
    </div>
    """, unsafe_allow_html=True)

# Funnel
st.markdown(f"#### 📊 Revenue Funnel — {selected_state}")
pop = state_row['Adult_Pop']
adj_p = max(0.01, min(0.60, state_row['Mean_Prevalence'] * (1 + prevalence_adj / 100)))
diab = pop * adj_p
diag = diab * (diagnosis_rate / 100)
acc  = diag * (market_access / 100)
addr = acc * (market_share / 100)
treat = addr * (adherence / 100)

fig_funnel = go.Figure(go.Funnel(
    y=['Adult Pop', 'Diabetic', 'Diagnosed', 'Accessible', 'Addressable', 'Treated'],
    x=[pop, diab, diag, acc, addr, treat],
    textinfo='text',
    text=[
        f"{pop/1e6:,.2f}M<br>(100.00%)",
        f"{diab/1e6:,.2f}M<br>({diab/pop*100:.2f}%)",
        f"{diag/1e6:,.2f}M<br>({diag/pop*100:.2f}%)",
        f"{acc/1e6:,.2f}M<br>({acc/pop*100:.2f}%)",
        f"{addr/1e3:,.1f}K<br>({addr/pop*100:.3f}%)",
        f"{treat/1e3:,.1f}K<br>({treat/pop*100:.3f}%)",
    ],
    marker=dict(color=['#3B82F6', '#8B5CF6', '#A78BFA', '#F59E0B', '#FBBF24', '#10B981']),
    hovertemplate='<b>%{y}</b><br>Patients: %{x:,.0f}<br>%{text}<extra></extra>'
))
# ============================================================
# SECTION 7: EXPORT
# ============================================================
st.markdown("<div class='section-header'>📥 Export Your Scenario</div>", unsafe_allow_html=True)

export_df = scenario_df.copy()
export_df['scenario_name']     = "Custom" if is_custom else "Base"
export_df['prevalence_adj']    = prevalence_adj
export_df['market_access']     = market_access
export_df['diagnosis_rate']    = diagnosis_rate
export_df['price_per_patient'] = price_per_patient
export_df['adherence']         = adherence
export_df['market_share']      = market_share

csv = export_df.to_csv(index=False).encode('utf-8')

e1, e2 = st.columns(2)
with e1:
    st.download_button("📥 Download Scenario CSV", data=csv,
        file_name=f"scenario_{prevalence_adj}_{market_access}_{price_per_patient}.csv",
        mime='text/csv', use_container_width=True)

with e2:
    summary = f"""SCENARIO SUMMARY
================
Author: Mohammad Arkam
Generated: India Pharma Forecasting Platform

PARAMETERS
==========
Prevalence Adjustment: {prevalence_adj:+d}%
Market Access:         {market_access}%
Diagnosis Rate:        {diagnosis_rate}%
Price per Patient:     ₹{price_per_patient:,}
Adherence:             {adherence}%
Market Share:          {market_share}%

RESULTS
=======
Total Revenue:    ₹{total_revenue:,.0f} Cr
Treated Patients: {total_patients/1e6:.2f}M
Revenue/Patient:  ₹{revenue_per_pat:,.0f}
vs Baseline:      {pct_change:+.1f}%

TOP 3 STATES BY REVENUE
=======================
"""
    for i, row in enumerate(scenario_df.nlargest(3, 'Revenue_Cr').itertuples(), 1):
        summary += f"{i}. {row.State}: ₹{row.Revenue_Cr:,.1f} Cr\n"
    
    st.download_button("📄 Download Summary (TXT)", data=summary,
        file_name="scenario_summary.txt", mime='text/plain',
        use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #475569; padding: 10px; font-size: 12px;'>
    🎯 Sensitivity Analysis • Bayesian-informed prevalence • Real-time recalculation
    <br>
    Data Sources: IHME GBD 2023 • ICMR-INDIAB-17 • NFHS-5 • UN World Population Prospects
</div>
""", unsafe_allow_html=True)