"""
🗺️ STATE-LEVEL EXPLORER
========================
Interactive state-by-state analysis:
- Priority rankings & tier classification
- Per-state forecast charts
- Hospital density correlation
- Top opportunity identification
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
    page_title="State Explorer",
    page_icon="🗺️",
    layout="wide"
)

# Custom CSS
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
    .tier-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .tier-1 { background-color: #DC2626; color: white; }
    .tier-2 { background-color: #EA580C; color: white; }
    .tier-3 { background-color: #F59E0B; color: white; }
    .tier-4 { background-color: #10B981; color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_state_master():
    return pd.read_csv("data/state_master.csv")

@st.cache_data
def load_state_forecast():
    return pd.read_csv("outputs/forecast_combined_state_year.csv")

@st.cache_data
def load_bayesian():
    return pd.read_csv("outputs/bayesian_state_intervals.csv")

df_master = load_state_master()
df_forecast = load_state_forecast()
df_bayesian = load_bayesian()

# ============================================================
# HEADER
# ============================================================
st.title("🗺️ State-Level Explorer")
st.markdown("**Interactive state-by-state analysis with priority rankings**")
st.markdown("---")

# ============================================================
# SIDEBAR FILTERS
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    
    # Region filter
    if 'Region' in df_master.columns:
        regions = ['All'] + sorted(df_master['Region'].unique().tolist())
        selected_region = st.selectbox("Region", regions)
    else:
        selected_region = 'All'
    
    # Tier filter
    tier_col = 'Market_Tier' if 'Market_Tier' in df_master.columns else None
    if tier_col:
        tiers = ['All'] + sorted(df_master[tier_col].unique().tolist())
        selected_tier = st.selectbox("Market Tier", tiers)
    else:
        selected_tier = 'All'
    
    # Top N filter
    top_n = st.slider("Show Top N States", min_value=5, max_value=31, value=10)
    
    # Sort by
    sort_options = {
        'Priority_Score': 'Priority Score',
        'Total_Diabetics': 'Total Diabetics',
        'Revenue_Cr': 'Revenue',
        'Opportunity_Revenue_Cr': 'Untapped Opportunity',
        'Diabetes_Prevalence_2021': 'Prevalence Rate'
    }
    available_sorts = {k: v for k, v in sort_options.items() if k in df_master.columns}
    sort_by = st.selectbox(
        "Sort by",
        options=list(available_sorts.keys()),
        format_func=lambda x: available_sorts[x]
    )

# Apply filters
df_filtered = df_master.copy()
if selected_region != 'All' and 'Region' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Region'] == selected_region]
if selected_tier != 'All' and tier_col:
    df_filtered = df_filtered[df_filtered[tier_col] == selected_tier]

df_filtered = df_filtered.sort_values(sort_by, ascending=False).head(top_n)

# ============================================================
# TOP-LEVEL METRICS
# ============================================================
total_diabetics = df_master['Total_Diabetics'].sum() / 1e6 if 'Total_Diabetics' in df_master.columns else 0
total_revenue = df_master['Revenue_Cr'].sum() if 'Revenue_Cr' in df_master.columns else 0
total_opportunity = df_master['Opportunity_Revenue_Cr'].sum() if 'Opportunity_Revenue_Cr' in df_master.columns else 0
n_states = len(df_master)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total States", f"{n_states}")
with m2:
    st.metric("Total Diabetics", f"{total_diabetics:.1f}M")
with m3:
    st.metric("Total Revenue", f"₹{total_revenue:,.0f} Cr")
with m4:
    st.metric("Untapped Opportunity", f"₹{total_opportunity:,.0f} Cr", delta="🎯 Growth potential")

# ============================================================
# PRIORITY RANKING TABLE
# ============================================================
st.markdown("<div class='section-header'>🏆 State Priority Rankings</div>", unsafe_allow_html=True)

# Build display table
display_cols = ['State']
if 'Region' in df_filtered.columns:
    display_cols.append('Region')
if 'Diabetes_Prevalence_2021' in df_filtered.columns:
    display_cols.append('Diabetes_Prevalence_2021')
if 'Total_Diabetics' in df_filtered.columns:
    display_cols.append('Total_Diabetics')
if 'Revenue_Cr' in df_filtered.columns:
    display_cols.append('Revenue_Cr')
if 'Opportunity_Revenue_Cr' in df_filtered.columns:
    display_cols.append('Opportunity_Revenue_Cr')
if 'Market_Tier' in df_filtered.columns:
    display_cols.append('Market_Tier')
if 'Priority_Score' in df_filtered.columns:
    display_cols.append('Priority_Score')

display_table = df_filtered[display_cols].copy().reset_index(drop=True)
display_table.index = display_table.index + 1
display_table.index.name = 'Rank'

# Format numbers
if 'Diabetes_Prevalence_2021' in display_table.columns:
    display_table['Diabetes_Prevalence_2021'] = (display_table['Diabetes_Prevalence_2021'] * 100).round(2).astype(str) + '%'
if 'Total_Diabetics' in display_table.columns:
    display_table['Total_Diabetics'] = (display_table['Total_Diabetics'] / 1e6).round(2).astype(str) + 'M'
if 'Revenue_Cr' in display_table.columns:
    display_table['Revenue_Cr'] = '₹' + display_table['Revenue_Cr'].round(1).astype(str) + ' Cr'
if 'Opportunity_Revenue_Cr' in display_table.columns:
    display_table['Opportunity_Revenue_Cr'] = '₹' + display_table['Opportunity_Revenue_Cr'].round(1).astype(str) + ' Cr'
if 'Priority_Score' in display_table.columns:
    display_table['Priority_Score'] = display_table['Priority_Score'].round(2)

# Rename columns for display
column_rename = {
    'Diabetes_Prevalence_2021': 'Prevalence',
    'Total_Diabetics': 'Diabetics',
    'Revenue_Cr': 'Revenue',
    'Opportunity_Revenue_Cr': 'Opportunity',
    'Market_Tier': 'Tier',
    'Priority_Score': 'Score'
}
display_table = display_table.rename(columns=column_rename)

st.dataframe(display_table, use_container_width=True, height=400)

# ============================================================
# DUAL CHARTS: BAR + SCATTER
# ============================================================
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("<div class='section-header'>📊 Top States by Priority</div>", unsafe_allow_html=True)
    
    if 'Priority_Score' in df_filtered.columns:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_filtered['Priority_Score'],
            y=df_filtered['State'],
            orientation='h',
            marker=dict(
                color=df_filtered['Priority_Score'],
                colorscale='Viridis',
                showscale=False
            ),
            text=df_filtered['Priority_Score'].round(1),
            textposition='outside'
        ))
        fig_bar.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_title='Priority Score',
            yaxis_title='',
            yaxis=dict(autorange='reversed'),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown("<div class='section-header'>🎯 Opportunity vs Current Revenue</div>", unsafe_allow_html=True)
    
    if 'Revenue_Cr' in df_filtered.columns and 'Opportunity_Revenue_Cr' in df_filtered.columns:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=df_filtered['Revenue_Cr'],
            y=df_filtered['Opportunity_Revenue_Cr'],
            mode='markers+text',
            text=df_filtered['State'],
            textposition='top center',
            textfont=dict(size=10, color='#CBD5E1'),
            marker=dict(
                size=df_filtered['Total_Diabetics'] / 1e5 if 'Total_Diabetics' in df_filtered.columns else 15,
                color=df_filtered['Priority_Score'] if 'Priority_Score' in df_filtered.columns else '#3B82F6',
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Priority<br>Score'),
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>%{text}</b><br>Revenue: ₹%{x:,.0f} Cr<br>Opportunity: ₹%{y:,.0f} Cr<extra></extra>'
        ))
        
        # Add diagonal reference line
        max_val = max(df_filtered['Revenue_Cr'].max(), df_filtered['Opportunity_Revenue_Cr'].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', dash='dash'),
            name='1:1 line',
            showlegend=False
        ))
        
        fig_scatter.update_layout(
            template='plotly_dark',
            height=500,
            xaxis_title='Current Revenue (₹ Cr)',
            yaxis_title='Untapped Opportunity (₹ Cr)',
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# INDIVIDUAL STATE DEEP-DIVE
# ============================================================
st.markdown("<div class='section-header'>🔍 Individual State Deep-Dive</div>", unsafe_allow_html=True)

# State selector
all_states = sorted(df_master['State'].unique().tolist())
default_state = 'Maharashtra' if 'Maharashtra' in all_states else all_states[0]
selected_state = st.selectbox("Select a state to explore", all_states, index=all_states.index(default_state))

# Get state info
state_info = df_master[df_master['State'] == selected_state].iloc[0]
state_forecast = df_forecast[df_forecast['State'] == selected_state] if 'State' in df_forecast.columns else None

# State Header Stats
sc1, sc2, sc3, sc4, sc5 = st.columns(5)

with sc1:
    if 'Region' in state_info:
        st.metric("Region", state_info['Region'])
with sc2:
    if 'Diabetes_Prevalence_2021' in state_info:
        st.metric("Prevalence", f"{state_info['Diabetes_Prevalence_2021']*100:.2f}%")
with sc3:
    if 'Total_Diabetics' in state_info:
        st.metric("Diabetics", f"{state_info['Total_Diabetics']/1e6:.2f}M")
with sc4:
    if 'Revenue_Cr' in state_info:
        st.metric("Current Revenue", f"₹{state_info['Revenue_Cr']:,.0f} Cr")
with sc5:
    if 'Priority_Score' in state_info:
        st.metric("Priority Score", f"{state_info['Priority_Score']:.1f}")

# Forecast chart for this state
if state_forecast is not None and len(state_forecast) > 0:
    st.markdown(f"#### 📈 {selected_state} — 20-Year Forecast")
    
    fig_state = go.Figure()
    
    # Detect demand column
    demand_col = None
    for col in ['State_Demand', 'Annual_Demand', 'State_Patients', 'Demand']:
        if col in state_forecast.columns:
            demand_col = col
            break
    
    revenue_col = None
    for col in ['State_Revenue_Cr', 'Revenue_Cr', 'State_Revenue']:
        if col in state_forecast.columns:
            revenue_col = col
            break
    
    if demand_col:
        fig_state.add_trace(go.Scatter(
            x=state_forecast['Year'],
            y=state_forecast[demand_col],
            mode='lines+markers',
            name=demand_col.replace('_', ' '),
            line=dict(color='#10B981', width=3),
            marker=dict(size=8),
            yaxis='y'
        ))
    
    if revenue_col:
        fig_state.add_trace(go.Scatter(
            x=state_forecast['Year'],
            y=state_forecast[revenue_col],
            mode='lines+markers',
            name='Revenue (₹ Cr)',
            line=dict(color='#F59E0B', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            yaxis='y2'
        ))
    
    fig_state.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Year',
        yaxis=dict(title='Patients / Demand', side='left'),
        yaxis2=dict(title='Revenue (₹ Cr)', overlaying='y', side='right'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=20, b=80)
    )
    
    st.plotly_chart(fig_state, use_container_width=True)

# Bayesian uncertainty (if available)
state_bayes = df_bayesian[df_bayesian['State'] == selected_state] if 'State' in df_bayesian.columns else None
if state_bayes is not None and len(state_bayes) > 0:
    st.markdown(f"#### 🔬 {selected_state} — Bayesian Uncertainty")
    
    bayes_row = state_bayes.iloc[0]
    bc1, bc2, bc3 = st.columns(3)
    
    with bc1:
        if 'Mean_Prevalence' in bayes_row:
            st.metric("Mean Prevalence", f"{bayes_row['Mean_Prevalence']*100:.2f}%")
    with bc2:
        if 'Lower_95_CI' in bayes_row and 'Upper_95_CI' in bayes_row:
            st.metric(
                "95% Credible Interval",
                f"{bayes_row['Lower_95_CI']*100:.2f}% — {bayes_row['Upper_95_CI']*100:.2f}%"
            )
    with bc3:
        if 'Uncertainty' in bayes_row:
            st.metric("Uncertainty Width", f"{bayes_row['Uncertainty']*100:.2f}%")

# ============================================================
# REGIONAL COMPARISON
# ============================================================
st.markdown("<div class='section-header'>🌍 Regional Comparison</div>", unsafe_allow_html=True)

if 'Region' in df_master.columns:
    region_agg = df_master.groupby('Region').agg({
        'Total_Diabetics': 'sum',
        'Revenue_Cr': 'sum',
        'Opportunity_Revenue_Cr': 'sum'
    }).reset_index()
    
    fig_region = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Diabetics by Region', 'Revenue by Region', 'Opportunity by Region'),
        specs=[[{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}]]
    )
    
    fig_region.add_trace(
        go.Pie(
            labels=region_agg['Region'],
            values=region_agg['Total_Diabetics'],
            hole=0.5,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set2)
        ), row=1, col=1
    )
    fig_region.add_trace(
        go.Pie(
            labels=region_agg['Region'],
            values=region_agg['Revenue_Cr'],
            hole=0.5,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set2)
        ), row=1, col=2
    )
    fig_region.add_trace(
        go.Pie(
            labels=region_agg['Region'],
            values=region_agg['Opportunity_Revenue_Cr'],
            hole=0.5,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set2)
        ), row=1, col=3
    )
    
    fig_region.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    st.plotly_chart(fig_region, use_container_width=True)

# ============================================================
# DOWNLOAD
# ============================================================
st.markdown("---")
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download State Data (CSV)",
    data=csv,
    file_name=f'state_explorer_{selected_region}_{selected_tier}.csv',
    mime='text/csv'
)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; padding: 10px; font-size: 12px;'>
    Data: 31 Indian states • ICMR-INDIAB-17 • NFHS-5 • Hospital Directory (30,273 hospitals)
</div>
""", unsafe_allow_html=True)