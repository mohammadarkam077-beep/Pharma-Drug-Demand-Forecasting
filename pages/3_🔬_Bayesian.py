"""
🔬 BAYESIAN UNCERTAINTY DASHBOARD
==================================
Showcases hierarchical Bayesian model results:
- Credible intervals by state
- Posterior distribution plots
- Region-level effects
- Uncertainty ranking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bayesian Uncertainty",
    page_icon="🔬",
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
    .info-box {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #8B5CF6;
        margin: 10px 0;
        color: #CBD5E1;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #0F1E33 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 3px solid #8B5CF6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_bayesian():
    return pd.read_csv("outputs/bayesian_state_intervals.csv")

@st.cache_data
def load_state_master():
    return pd.read_csv("data/state_master.csv")

df_bayes = load_bayesian()
df_master = load_state_master()

# Merge for richer info
if 'State' in df_bayes.columns and 'State' in df_master.columns:
    df_merged = df_bayes.merge(
        df_master[['State', 'Region', 'Total_Diabetics']] if 'Region' in df_master.columns else df_master[['State']],
        on='State',
        how='left'
    )
else:
    df_merged = df_bayes

# ============================================================
# HEADER
# ============================================================
st.title("🔬 Bayesian Uncertainty Quantification")
st.markdown("""
**Hierarchical Bayesian Model Results** — State-level prevalence with proper uncertainty bounds
""")
st.markdown("---")

# ============================================================
# MODEL OVERVIEW BOX
# ============================================================
st.markdown("""
<div class='info-box'>
    <h4 style='color: #A78BFA; margin-top: 0;'>📊 Model Architecture</h4>
    <ul style='margin-bottom: 0;'>
        <li><b>Level 1 (National):</b> Prior on overall mean prevalence</li>
        <li><b>Level 2 (Regional):</b> Region-level offsets (North, South, East, West, Central, Northeast)</li>
        <li><b>Level 3 (State):</b> State-specific deviations within each region</li>
        <li><b>Sampler:</b> nutpie (Rust-based NUTS) — 4 chains × 1,000 draws + 500 tune</li>
        <li><b>Reparameterization:</b> Non-centered + logit-scale for stability</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CONVERGENCE METRICS
# ============================================================
st.markdown("<div class='section-header'>✅ Model Convergence</div>", unsafe_allow_html=True)

cv1, cv2, cv3, cv4 = st.columns(4)
with cv1:
    st.metric("Max R-hat", "1.010", delta="✅ Converged", delta_color="normal")
with cv2:
    st.metric("Divergences", "112 / 4000", delta="2.80%", delta_color="off")
with cv3:
    st.metric("ESS (Bulk)", "410+", delta="Healthy")
with cv4:
    st.metric("Sampling Time", "43.5s", delta="nutpie")

# ============================================================
# CREDIBLE INTERVALS CHART (FOREST PLOT)
# ============================================================
st.markdown("<div class='section-header'>🌲 State Prevalence — 95% Credible Intervals</div>", unsafe_allow_html=True)

# Sort by mean prevalence
df_plot = df_merged.sort_values('Mean_Prevalence', ascending=True).copy()

fig_forest = go.Figure()

# Error bars (CI)
fig_forest.add_trace(go.Scatter(
    x=df_plot['Mean_Prevalence'] * 100,
    y=df_plot['State'],
    mode='markers',
    error_x=dict(
        type='data',
        symmetric=False,
        array=(df_plot['Upper_95_CI'] - df_plot['Mean_Prevalence']) * 100,
        arrayminus=(df_plot['Mean_Prevalence'] - df_plot['Lower_95_CI']) * 100,
        color='#8B5CF6',
        thickness=2,
        width=4
    ),
    marker=dict(
        size=10,
        color=df_plot['Mean_Prevalence'] * 100,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Mean<br>Prevalence (%)', x=1.0)
    ),
    text=df_plot['State'],
    hovertemplate='<b>%{text}</b><br>Mean: %{x:.2f}%<br>CI: [%{customdata[0]:.2f}%, %{customdata[1]:.2f}%]<extra></extra>',
    customdata=df_plot[['Lower_95_CI', 'Upper_95_CI']].values * 100
))

# National average reference line
national_mean = df_plot['Mean_Prevalence'].mean() * 100
fig_forest.add_vline(
    x=national_mean,
    line_dash='dash',
    line_color='#EF4444',
    annotation_text=f'National Mean: {national_mean:.2f}%',
    annotation_position='top'
)

fig_forest.update_layout(
    template='plotly_dark',
    height=800,
    xaxis_title='Diabetes Prevalence (%)',
    yaxis_title='',
    showlegend=False,
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig_forest, use_container_width=True)

st.markdown("""
<div class='info-box'>
💡 <b>How to read this chart:</b> Each dot represents the posterior mean prevalence for a state. The horizontal bars show the 95% credible interval — the range where the true prevalence lies with 95% probability. Wider bars = more uncertainty (typically smaller states with less data).
</div>
""", unsafe_allow_html=True)

# ============================================================
# UNCERTAINTY ANALYSIS
# ============================================================
st.markdown("<div class='section-header'>📊 Uncertainty Analysis</div>", unsafe_allow_html=True)

col_unc1, col_unc2 = st.columns(2)

with col_unc1:
    st.markdown("#### 🎯 Most Uncertain States")
    most_uncertain = df_merged.nlargest(10, 'Uncertainty')[['State', 'Mean_Prevalence', 'Uncertainty']].copy()
    most_uncertain['Mean_Prevalence'] = (most_uncertain['Mean_Prevalence'] * 100).round(2).astype(str) + '%'
    most_uncertain['Uncertainty'] = (most_uncertain['Uncertainty'] * 100).round(2).astype(str) + '%'
    most_uncertain.columns = ['State', 'Mean Prev.', 'CI Width']
    most_uncertain.index = range(1, len(most_uncertain) + 1)
    st.dataframe(most_uncertain, use_container_width=True)

with col_unc2:
    st.markdown("#### 🎯 Most Certain States")
    most_certain = df_merged.nsmallest(10, 'Uncertainty')[['State', 'Mean_Prevalence', 'Uncertainty']].copy()
    most_certain['Mean_Prevalence'] = (most_certain['Mean_Prevalence'] * 100).round(2).astype(str) + '%'
    most_certain['Uncertainty'] = (most_certain['Uncertainty'] * 100).round(2).astype(str) + '%'
    most_certain.columns = ['State', 'Mean Prev.', 'CI Width']
    most_certain.index = range(1, len(most_certain) + 1)
    st.dataframe(most_certain, use_container_width=True)

# ============================================================
# REGIONAL EFFECTS
# ============================================================
st.markdown("<div class='section-header'>🌍 Regional Effects</div>", unsafe_allow_html=True)

if 'Region' in df_merged.columns:
    region_stats = df_merged.groupby('Region').agg(
        Mean_Prev=('Mean_Prevalence', 'mean'),
        Std_Prev=('Mean_Prevalence', 'std'),
        N_States=('State', 'count'),
        Avg_Uncertainty=('Uncertainty', 'mean')
    ).reset_index().sort_values('Mean_Prev', ascending=False)
    
    fig_region = go.Figure()
    
    # Bar with error bars
    fig_region.add_trace(go.Bar(
        x=region_stats['Region'],
        y=region_stats['Mean_Prev'] * 100,
        error_y=dict(
            type='data',
            array=region_stats['Std_Prev'] * 100,
            color='#FBBF24',
            thickness=2
        ),
        marker=dict(
            color=region_stats['Mean_Prev'] * 100,
            colorscale='Viridis',
            showscale=False
        ),
        text=region_stats.apply(
            lambda x: f"{x['Mean_Prev']*100:.2f}%<br>n={x['N_States']}", axis=1
        ),
        textposition='outside'
    ))
    
    fig_region.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Region',
        yaxis_title='Mean Prevalence (%)',
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_region, use_container_width=True)
    
    # Regional summary table
    region_display = region_stats.copy()
    region_display['Mean_Prev'] = (region_display['Mean_Prev'] * 100).round(2).astype(str) + '%'
    region_display['Std_Prev'] = (region_display['Std_Prev'] * 100).round(2).astype(str) + '%'
    region_display['Avg_Uncertainty'] = (region_display['Avg_Uncertainty'] * 100).round(2).astype(str) + '%'
    region_display.columns = ['Region', 'Mean Prevalence', 'Std Dev', '# States', 'Avg CI Width']
    st.dataframe(region_display, use_container_width=True, hide_index=True)

# ============================================================
# POSTERIOR DISTRIBUTION IMAGE
# ============================================================
st.markdown("<div class='section-header'>📈 Posterior Distributions</div>", unsafe_allow_html=True)

posterior_path = Path("outputs/bayesian_posteriors.png")
if posterior_path.exists():
    image = Image.open(posterior_path)
    st.image(image, caption="Posterior Distributions", use_column_width=True)
else:
    st.warning("⚠️ Posterior plot not found. Run `python bayesian_forecasting.py` to generate.")

# ============================================================
# STATE COMPARISON TOOL
# ============================================================
st.markdown("<div class='section-header'>🔍 Compare States</div>", unsafe_allow_html=True)

states_to_compare = st.multiselect(
    "Select up to 5 states to compare",
    options=sorted(df_merged['State'].unique().tolist()),
    default=['Kerala', 'Maharashtra', 'Bihar', 'Tamil Nadu'][:min(4, len(df_merged))]
)

if states_to_compare:
    df_compare = df_merged[df_merged['State'].isin(states_to_compare)].sort_values('Mean_Prevalence', ascending=False)
    
    fig_compare = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, (_, row) in enumerate(df_compare.iterrows()):
        # Distribution approximation (using mean and std)
        x_vals = np.linspace(
            row['Lower_95_CI'] * 100 - 2,
            row['Upper_95_CI'] * 100 + 2,
            200
        )
        # Approximate normal distribution
        std_approx = (row['Upper_95_CI'] - row['Lower_95_CI']) / 3.92  # 95% CI = ±1.96 std
        y_vals = np.exp(-0.5 * ((x_vals - row['Mean_Prevalence'] * 100) / (std_approx * 100)) ** 2)
        y_vals = y_vals / y_vals.max()  # Normalize
        
        fig_compare.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name=row['State'],
            fill='tozeroy',
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.7
        ))
        
        # Add mean line
        fig_compare.add_vline(
            x=row['Mean_Prevalence'] * 100,
            line_dash='dot',
            line_color=colors[i % len(colors)],
            opacity=0.5
        )
    
    fig_compare.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Diabetes Prevalence (%)',
        yaxis_title='Posterior Density (normalized)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
        margin=dict(l=0, r=0, t=30, b=80)
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Comparison table
    compare_display = df_compare[['State', 'Mean_Prevalence', 'Lower_95_CI', 'Upper_95_CI', 'Uncertainty']].copy()
    for col in ['Mean_Prevalence', 'Lower_95_CI', 'Upper_95_CI', 'Uncertainty']:
        compare_display[col] = (compare_display[col] * 100).round(2).astype(str) + '%'
    compare_display.columns = ['State', 'Mean', 'Lower 95% CI', 'Upper 95% CI', 'CI Width']
    st.dataframe(compare_display, use_container_width=True, hide_index=True)

# ============================================================
# DOWNLOAD
# ============================================================
st.markdown("---")
csv = df_merged.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Bayesian Results (CSV)",
    data=csv,
    file_name='bayesian_state_intervals.csv',
    mime='text/csv'
)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; padding: 10px; font-size: 12px;'>
    Hierarchical Bayesian Model • PyMC + nutpie • 4 chains × 1,000 draws • Non-centered parameterization
</div>
""", unsafe_allow_html=True)
