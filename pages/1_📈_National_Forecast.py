"""
📈 NATIONAL FORECAST DEEP-DIVE
================================
Detailed analysis of national-level forecasts:
- Multi-model comparison (Funnel, Monte Carlo, XGBoost, Ridge, Lasso)
- Patient acquisition funnel
- Year-over-year metrics
- Model accuracy
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="National Forecast",
    page_icon="📈",
    layout="wide"
)

# Custom CSS (reuse main app theme)
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
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #0F1E33 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 3px solid #3B82F6;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv("outputs/forecast_output_advanced.csv")

df = load_data()

# ============================================================
# HEADER
# ============================================================
st.title("📈 National Forecast Deep-Dive")
st.markdown("**Detailed multi-model forecasting analysis with uncertainty quantification**")
st.markdown("---")

# ============================================================
# YEAR RANGE FILTER (Sidebar)
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    year_range = st.slider(
        "Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max())),
        step=1
    )
    
    show_models = st.multiselect(
        "Models to display",
        options=['Annual_Demand', 'Demand_Mean', 'XGB_Predicted', 'Lasso_Predicted', 'Ensemble_Forecast'],
        default=['Annual_Demand', 'Demand_Mean', 'Ensemble_Forecast'],
        format_func=lambda x: {
            'Annual_Demand': 'Deterministic Funnel',
            'Demand_Mean': 'Monte Carlo (10K)',
            'XGB_Predicted': 'XGBoost ML',
            'Lasso_Predicted': 'Lasso ML',
            'Ensemble_Forecast': 'Ensemble'
        }.get(x, x)
    )
    
    show_confidence = st.checkbox("Show Confidence Bands", value=True)

# Filter data
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# ============================================================
# KEY METRICS ROW
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_demand = df_filtered['Annual_Demand'].sum() / 1e9
    st.metric("Total Demand (Period)", f"{total_demand:,.1f}B", "units")

with col2:
    avg_growth = df_filtered['Annual_Demand'].pct_change().mean() * 100
    st.metric("Avg YoY Growth", f"{avg_growth:.1f}%", "annual")

with col3:
    total_revenue = df_filtered['Revenue_Cr'].sum()
    st.metric("Total Revenue (Period)", f"₹{total_revenue:,.0f} Cr")

with col4:
    peak_demand_year = df_filtered.loc[df_filtered['Annual_Demand'].idxmax(), 'Year']
    st.metric("Peak Demand Year", f"{int(peak_demand_year)}")

# ============================================================
# MULTI-MODEL COMPARISON CHART
# ============================================================
st.markdown("<div class='section-header'>🔬 Multi-Model Comparison</div>", unsafe_allow_html=True)

fig = go.Figure()

model_colors = {
    'Annual_Demand': '#10B981',      # Green - deterministic
    'Demand_Mean': '#8B5CF6',         # Purple - Monte Carlo
    'XGB_Predicted': '#F59E0B',       # Orange - XGBoost
    'Lasso_Predicted': '#3B82F6',     # Blue - Lasso
    'Ensemble_Forecast': '#EF4444'    # Red - Ensemble
}

model_labels = {
    'Annual_Demand': 'Deterministic Funnel',
    'Demand_Mean': 'Monte Carlo Mean',
    'XGB_Predicted': 'XGBoost ML',
    'Lasso_Predicted': 'Lasso ML',
    'Ensemble_Forecast': 'Ensemble Forecast'
}

# Confidence band first (if enabled)
if show_confidence:
    fig.add_trace(go.Scatter(
        x=df_filtered['Year'].tolist() + df_filtered['Year'].tolist()[::-1],
        y=(df_filtered['Demand_P95']/1e9).tolist() + (df_filtered['Demand_P5']/1e9).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% CI (Monte Carlo)',
        hoverinfo='skip'
    ))

# Add selected models
for model in show_models:
    if model in df_filtered.columns:
        fig.add_trace(go.Scatter(
            x=df_filtered['Year'],
            y=df_filtered[model]/1e9,
            mode='lines+markers',
            name=model_labels.get(model, model),
            line=dict(color=model_colors.get(model, '#FFFFFF'), width=2.5),
            marker=dict(size=7)
        ))

fig.update_layout(
    template='plotly_dark',
    height=500,
    hovermode='x unified',
    xaxis_title='Year',
    yaxis_title='Annual Demand (Billion Units)',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=-0.25,
        xanchor='center',
        x=0.5,
        bgcolor='rgba(0,0,0,0)'
    ),
    margin=dict(l=0, r=0, t=30, b=80)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PATIENT FUNNEL VISUALIZATION
# ============================================================
st.markdown("<div class='section-header'>🔻 Patient Acquisition Funnel</div>", unsafe_allow_html=True)

# Year selector for funnel
funnel_year = st.select_slider(
    "Select Year",
    options=sorted(df_filtered['Year'].unique().astype(int).tolist()),
    value=int(df_filtered[df_filtered['Is_Forecast'] == 0]['Year'].max()) if (df_filtered['Is_Forecast'] == 0).any() else int(df_filtered['Year'].max())
)

selected = df_filtered[df_filtered['Year'] == funnel_year].iloc[0]

# Build funnel stages
funnel_stages = {
    'Total Population': selected['Population'],
    'Eligible Adults': selected['Eligible_Pop'],
    'Diabetic Pool': selected['Eligible_Pop'] * selected['prevalence'],
    'Diagnosed': selected['Eligible_Pop'] * selected['prevalence'] * selected['diagnosis_rate'],
    'On Treatment': selected['Eligible_Pop'] * selected['prevalence'] * selected['diagnosis_rate'] * selected['treatment_rate'],
    'Our Patients': selected['Patients']
}

funnel_df = pd.DataFrame({
    'Stage': list(funnel_stages.keys()),
    'Count_M': [v/1e6 for v in funnel_stages.values()]
})

# Calculate conversion rates
funnel_df['Conversion'] = (funnel_df['Count_M'] / funnel_df['Count_M'].iloc[0] * 100).round(2)

col_funnel, col_metrics = st.columns([3, 1])

with col_funnel:
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_df['Stage'],
        x=funnel_df['Count_M'],
        textinfo="value+percent initial",
        texttemplate='<b>%{value:.1f}M</b><br>(%{percentInitial})',
        marker=dict(
            color=['#3B82F6', '#06B6D4', '#10B981', '#84CC16', '#F59E0B', '#EF4444'],
            line=dict(width=2, color='#1E293B')
        ),
        connector=dict(line=dict(color='#475569', width=2))
    ))
    
    fig_funnel.update_layout(
        template='plotly_dark',
        height=500,
        title=f"Patient Funnel — {funnel_year}",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    st.plotly_chart(fig_funnel, use_container_width=True)

with col_metrics:
    st.markdown("### Key Rates")
    st.metric("Prevalence", f"{selected['prevalence']*100:.2f}%")
    st.metric("Diagnosis Rate", f"{selected['diagnosis_rate']*100:.1f}%")
    st.metric("Treatment Rate", f"{selected['treatment_rate']*100:.1f}%")
    st.metric("Market Share", f"{selected['market_share']*100:.1f}%")
    st.metric("Compliance", f"{selected['compliance']*100:.1f}%")

# ============================================================
# YEAR-OVER-YEAR ANALYSIS
# ============================================================
st.markdown("<div class='section-header'>📊 Year-Over-Year Metrics</div>", unsafe_allow_html=True)

# Calculate YoY changes
df_yoy = df_filtered.copy()
df_yoy['Demand_YoY'] = df_yoy['Annual_Demand'].pct_change() * 100
df_yoy['Revenue_YoY'] = df_yoy['Revenue_Cr'].pct_change() * 100
df_yoy['Patients_YoY'] = df_yoy['Patients'].pct_change() * 100

# Two subplots
fig_yoy = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Demand Growth (%)', 'Revenue Growth (%)'),
    horizontal_spacing=0.15
)

fig_yoy.add_trace(
    go.Bar(
        x=df_yoy['Year'][1:],
        y=df_yoy['Demand_YoY'][1:],
        marker_color=['#10B981' if x > 0 else '#EF4444' for x in df_yoy['Demand_YoY'][1:]],
        text=df_yoy['Demand_YoY'][1:].round(1),
        texttemplate='%{text}%',
        textposition='outside',
        name='Demand'
    ), row=1, col=1
)

fig_yoy.add_trace(
    go.Bar(
        x=df_yoy['Year'][1:],
        y=df_yoy['Revenue_YoY'][1:],
        marker_color=['#3B82F6' if x > 0 else '#EF4444' for x in df_yoy['Revenue_YoY'][1:]],
        text=df_yoy['Revenue_YoY'][1:].round(1),
        texttemplate='%{text}%',
        textposition='outside',
        name='Revenue'
    ), row=1, col=2
)

fig_yoy.update_layout(
    template='plotly_dark',
    height=400,
    showlegend=False,
    margin=dict(l=0, r=0, t=50, b=0)
)

st.plotly_chart(fig_yoy, use_container_width=True)

# ============================================================
# DETAILED DATA TABLE
# ============================================================
st.markdown("<div class='section-header'>📋 Detailed Forecast Data</div>", unsafe_allow_html=True)

display_cols = [
    'Year', 'Population', 'prevalence', 'diagnosis_rate', 'treatment_rate',
    'compliance', 'market_share', 'Patients', 'Annual_Demand', 'Revenue_Cr',
    'Demand_Worst', 'Demand_Base', 'Demand_Best', 'Is_Forecast'
]
available_cols = [c for c in display_cols if c in df_filtered.columns]

display_df = df_filtered[available_cols].copy()

# Format numbers
display_df['Population'] = (display_df['Population'] / 1e6).round(1).astype(str) + 'M'
display_df['prevalence'] = (display_df['prevalence'] * 100).round(2).astype(str) + '%'
display_df['diagnosis_rate'] = (display_df['diagnosis_rate'] * 100).round(1).astype(str) + '%'
display_df['treatment_rate'] = (display_df['treatment_rate'] * 100).round(1).astype(str) + '%'
display_df['compliance'] = (display_df['compliance'] * 100).round(1).astype(str) + '%'
display_df['market_share'] = (display_df['market_share'] * 100).round(1).astype(str) + '%'
display_df['Patients'] = (display_df['Patients'] / 1e6).round(2).astype(str) + 'M'
display_df['Annual_Demand'] = (display_df['Annual_Demand'] / 1e9).round(2).astype(str) + 'B'
display_df['Demand_Worst'] = (display_df['Demand_Worst'] / 1e9).round(2).astype(str) + 'B'
display_df['Demand_Base'] = (display_df['Demand_Base'] / 1e9).round(2).astype(str) + 'B'
display_df['Demand_Best'] = (display_df['Demand_Best'] / 1e9).round(2).astype(str) + 'B'
display_df['Is_Forecast'] = display_df['Is_Forecast'].map({0: 'Historical', 1: 'Forecast'})

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================
# DOWNLOAD BUTTON
# ============================================================
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data (CSV)",
    data=csv,
    file_name=f'national_forecast_{year_range[0]}_{year_range[1]}.csv',
    mime='text/csv'
)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; padding: 10px; font-size: 12px;'>
    Models: Deterministic Funnel + Monte Carlo (10K) + ARIMA + Holt-Winters + XGBoost + Ridge + Lasso<br>
    Data: 15 years historical (2010-2024) + 5 years forecast (2025-2029)
</div>
""", unsafe_allow_html=True)