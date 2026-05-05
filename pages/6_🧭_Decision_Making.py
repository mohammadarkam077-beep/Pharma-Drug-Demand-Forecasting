# pages/6_🧭_Decision_Making.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from decision_engine import (
    build_decision_table,
    add_inventory_decision,
    optimize_investment_portfolio,
    run_scenario_simulation,
    run_targeted_state_scenario,
    find_required_improvement_for_next_action,
    generate_decision_alerts,
    add_abc_classification
)

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Decision Making",
    page_icon="🧭",
    layout="wide"
)


# -----------------------------
# Page title
# -----------------------------

st.title("🧭 Pharma Decision-Making Dashboard")

st.markdown(
    """
    This page converts your forecasting output into **business decisions**.

    Instead of only showing predicted demand, this page answers:

    - Which states should we invest in?
    - Which states should we pilot first?
    - Which states should we monitor?
    - Which states should we deprioritize?
    - How much stock should we prepare?
    """
)


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

forecast_path = BASE_DIR / "outputs" / "forecast_combined_state_year.csv"
hospital_path = BASE_DIR / "data" / "hospitals_by_state.csv"
bayesian_path = BASE_DIR / "outputs" / "bayesian_state_intervals.csv"
save_path = BASE_DIR / "outputs" / "decision_recommendations.csv"


# -----------------------------
# Load required data
# -----------------------------

if not forecast_path.exists():
    st.error(
        """
        `outputs/forecast_combined_state_year.csv` was not found.

        Please run this command first:

        ```bash
        python run_all.py
        ```

        Then refresh this Streamlit page.
        """
    )
    st.stop()

forecast_df = pd.read_csv(forecast_path)

hospital_df = None
if hospital_path.exists():
    hospital_df = pd.read_csv(hospital_path)

bayesian_df = None
if bayesian_path.exists():
    bayesian_df = pd.read_csv(bayesian_path)


# -----------------------------
# Build decision table
# -----------------------------

try:
    decision_df = build_decision_table(
        forecast_df=forecast_df,
        hospital_df=hospital_df,
        bayesian_df=bayesian_df
    )
except Exception as e:
    st.error("Decision engine failed.")
    st.exception(e)

    st.subheader("Forecast columns found")
    st.write(forecast_df.columns.tolist())

    st.stop()


if decision_df.empty:
    st.warning("Decision table is empty. Please check your forecast data.")
    st.stop()
decision_df = add_abc_classification(decision_df)


# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.header("Inventory Settings")

show_inventory = st.sidebar.checkbox(
    "Add inventory decision logic",
    value=True
)

stock_percentage = st.sidebar.slider(
    "Current stock as % of forecast demand",
    min_value=10,
    max_value=150,
    value=70,
    step=5
)

current_stock_factor = stock_percentage / 100

service_level = st.sidebar.selectbox(
    "Service level",
    [
        "90%",
        "95%",
        "98%"
    ],
    index=1
)

service_multiplier_map = {
    "90%": 1.28,
    "95%": 1.65,
    "98%": 2.05
}

service_multiplier = service_multiplier_map[service_level]


# -----------------------------
# Add inventory logic if selected
# -----------------------------

if show_inventory:
    decision_df["current_stock"] = (
        decision_df["forecast_demand"] * current_stock_factor
    )

    decision_df = add_inventory_decision(
        decision_df,
        current_stock=decision_df["current_stock"],
        service_level_multiplier=service_multiplier
    )


# -----------------------------
# ABC inventory classification
# -----------------------------

st.markdown("---")
st.subheader("ABC Inventory Classification")

st.markdown(
    """
    ABC classification groups states by forecast revenue contribution.

    - **A class:** highest revenue contribution, should receive highest service priority.
    - **B class:** medium revenue contribution, standard monitoring.
    - **C class:** lower revenue contribution, controlled inventory exposure.
    """
)

abc_display = decision_df.copy()

abc_display["revenue_share_pct"] = (
    abc_display["revenue_share"] * 100
).round(2)

abc_display["cumulative_revenue_share_pct"] = (
    abc_display["cumulative_revenue_share"] * 100
).round(2)

abc_display["forecast_revenue_crore"] = (
    abc_display["forecast_revenue"] / 10_000_000
).round(2)

abc_cols = [
    "state",
    "forecast_revenue_crore",
    "revenue_share_pct",
    "cumulative_revenue_share_pct",
    "abc_class",
    "recommended_action"
]

st.dataframe(
    abc_display[abc_cols],
    use_container_width=True,
    hide_index=True
)

abc_count = (
    abc_display["abc_class"]
    .value_counts()
    .reset_index()
)

abc_count.columns = ["abc_class", "count"]

fig_abc = px.bar(
    abc_count,
    x="abc_class",
    y="count",
    color="abc_class",
    title="ABC Class Distribution",
    labels={
        "abc_class": "ABC Class",
        "count": "Number of States"
    }
)

st.plotly_chart(fig_abc, use_container_width=True)


# -----------------------------
# Budget planning sidebar
# -----------------------------

st.sidebar.markdown("---")
st.sidebar.header("Budget Planning")

available_budget = st.sidebar.number_input(
    "Available investment budget ₹ Cr",
    min_value=1,
    max_value=10000,
    value=100,
    step=10
)


# -----------------------------
# Create display version with readable units
# -----------------------------

display_df = decision_df.copy()

display_df["forecast_demand_million"] = (
    display_df["forecast_demand"] / 1_000_000
).round(2)

display_df["forecast_revenue_crore"] = (
    display_df["forecast_revenue"] / 10_000_000
).round(2)

display_df["opportunity_score"] = display_df["opportunity_score"].round(2)
display_df["risk_score"] = display_df["risk_score"].round(2)
display_df["growth_rate"] = display_df["growth_rate"].round(2)

if "hospital_count" in display_df.columns:
    display_df["hospital_count"] = display_df["hospital_count"].fillna(0).round(0).astype(int)

if show_inventory:
    display_df["current_stock_million"] = (
        display_df["current_stock"] / 1_000_000
    ).round(2)

    display_df["safety_stock_million"] = (
        display_df["safety_stock"] / 1_000_000
    ).round(2)

    display_df["target_stock_million"] = (
        display_df["target_stock"] / 1_000_000
    ).round(2)

    display_df["recommended_order_quantity_million"] = (
        display_df["recommended_order_quantity"] / 1_000_000
    ).round(2)


# -----------------------------
# Executive KPI cards
# -----------------------------

st.subheader("Executive Decision Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total States",
        len(decision_df)
    )

with col2:
    st.metric(
        "Invest Aggressively",
        int((decision_df["recommended_action"] == "Invest Aggressively").sum())
    )

with col3:
    st.metric(
        "Pilot First",
        int((decision_df["recommended_action"] == "Pilot First").sum())
    )

with col4:
    st.metric(
        "Selective Expansion",
        int((decision_df["recommended_action"] == "Selective Expansion").sum())
    )

with col5:
    st.metric(
        "Deprioritize",
        int((decision_df["recommended_action"] == "Deprioritize").sum())
    )


# -----------------------------
# Top recommendation box
# -----------------------------

top_state = decision_df.sort_values("priority_rank").iloc[0]

st.success(
    f"""
    **Recommended first priority:** {top_state['state']}

    **Action:** {top_state['recommended_action']}

    **Reason:** {top_state['reason']}
    """
)


# -----------------------------
# Management alerts
# -----------------------------

st.markdown("---")
st.subheader("Management Alerts")

alerts_df = generate_decision_alerts(decision_df)

if alerts_df.empty:
    st.success("No major decision alerts detected.")
else:
    high_alerts = int((alerts_df["severity"] == "High").sum())
    medium_alerts = int((alerts_df["severity"] == "Medium").sum())

    alert_col1, alert_col2, alert_col3 = st.columns(3)

    with alert_col1:
        st.metric("Total Alerts", len(alerts_df))

    with alert_col2:
        st.metric("High Severity", high_alerts)

    with alert_col3:
        st.metric("Medium Severity", medium_alerts)

    st.markdown("### Alert Filters")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_severity = st.multiselect(
            "Filter by severity",
            options=sorted(alerts_df["severity"].unique()),
            default=sorted(alerts_df["severity"].unique())
        )

    with filter_col2:
        selected_alert_type = st.multiselect(
            "Filter by alert type",
            options=sorted(alerts_df["alert_type"].unique()),
            default=sorted(alerts_df["alert_type"].unique())
        )

    filtered_alerts = alerts_df[
        alerts_df["severity"].isin(selected_severity)
        & alerts_df["alert_type"].isin(selected_alert_type)
    ].copy()

    severity_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    filtered_alerts["severity_rank"] = (
        filtered_alerts["severity"]
        .map(severity_order)
        .fillna(99)
    )

    filtered_alerts = filtered_alerts.sort_values(
        ["severity_rank", "state"]
    )

    show_top_alerts = st.checkbox(
        "Show only top 10 alerts",
        value=True
    )

    if show_top_alerts:
        display_alerts = filtered_alerts.head(10)
    else:
        display_alerts = filtered_alerts

    st.dataframe(
        display_alerts.drop(columns=["severity_rank"]),
        use_container_width=True,
        hide_index=True
    )


# -----------------------------
# Decision logic explanation
# -----------------------------

st.info(
    """
    **Decision logic used:**

    - High opportunity + low risk = Invest Aggressively
    - High opportunity + high risk = Pilot First
    - Medium opportunity = Selective Expansion
    - Low-medium opportunity = Maintain and Monitor
    - Low opportunity = Deprioritize
    """
)


# -----------------------------
# Recommended business actions table
# -----------------------------

st.markdown("---")
st.subheader("Recommended Business Actions")

main_cols = [
    "priority_rank",
    "state",
    "year",
    "forecast_demand_million",
    "forecast_revenue_crore",
    "growth_rate",
    "hospital_count",
    "opportunity_score",
    "risk_score",
    "decision_confidence",
    "recommended_action",
    "reason"
]

available_main_cols = [
    col for col in main_cols
    if col in display_df.columns
]

st.dataframe(
    display_df[available_main_cols],
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Charts
# -----------------------------

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("Opportunity Score by State")

    chart_df = decision_df.sort_values("opportunity_score", ascending=True)

    fig = px.bar(
        chart_df,
        x="opportunity_score",
        y="state",
        color="recommended_action",
        orientation="h",
        title="State Priority Ranking",
        labels={
            "opportunity_score": "Opportunity Score",
            "state": "State",
            "recommended_action": "Recommended Action"
        }
    )

    fig.update_layout(
        height=650,
        yaxis_title="State",
        xaxis_title="Opportunity Score"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Decision Category Distribution")

    action_count = (
        decision_df["recommended_action"]
        .value_counts()
        .reset_index()
    )

    action_count.columns = ["recommended_action", "count"]

    fig2 = px.pie(
        action_count,
        names="recommended_action",
        values="count",
        title="Recommended Action Mix"
    )

    fig2.update_layout(height=650)

    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# Opportunity-risk decision matrix
# -----------------------------

st.markdown("---")
st.subheader("Opportunity-Risk Decision Matrix")

st.markdown(
    """
    This matrix compares states based on **business opportunity** and **forecast risk**.

    - High opportunity + low risk = Scale / Invest
    - High opportunity + high risk = Pilot / Validate
    - Low opportunity + low risk = Maintain
    - Low opportunity + high risk = Avoid / Deprioritize
    """
)

matrix_df = decision_df.copy()

fig_matrix = px.scatter(
    matrix_df,
    x="risk_score",
    y="opportunity_score",
    size="forecast_revenue",
    color="recommended_action",
    hover_name="state",
    title="Opportunity vs Risk Matrix",
    labels={
        "risk_score": "Risk Score",
        "opportunity_score": "Opportunity Score",
        "forecast_revenue": "Forecast Revenue",
        "recommended_action": "Recommended Action"
    }
)

fig_matrix.add_hline(
    y=0.50,
    line_dash="dash",
    line_color="gray"
)

fig_matrix.add_vline(
    x=0.50,
    line_dash="dash",
    line_color="gray"
)

fig_matrix.add_annotation(
    x=0.20,
    y=0.85,
    text="Scale / Invest",
    showarrow=False,
    font=dict(size=14, color="green")
)

fig_matrix.add_annotation(
    x=0.80,
    y=0.85,
    text="Pilot / Validate",
    showarrow=False,
    font=dict(size=14, color="orange")
)

fig_matrix.add_annotation(
    x=0.20,
    y=0.20,
    text="Maintain",
    showarrow=False,
    font=dict(size=14, color="lightblue")
)

fig_matrix.add_annotation(
    x=0.80,
    y=0.20,
    text="Avoid / Deprioritize",
    showarrow=False,
    font=dict(size=14, color="red")
)

fig_matrix.update_layout(
    height=650,
    xaxis_title="Risk Score",
    yaxis_title="Opportunity Score"
)

st.plotly_chart(fig_matrix, use_container_width=True)


# -----------------------------
# Scenario simulation
# -----------------------------

st.markdown("---")
st.subheader("What-If Scenario Simulation")

st.markdown(
    """
    Use this section to test how decisions change when demand, revenue, or risk assumptions change.
    """
)

scenario_col1, scenario_col2, scenario_col3 = st.columns(3)

with scenario_col1:
    demand_change = st.slider(
        "Demand change %",
        min_value=-50,
        max_value=100,
        value=0,
        step=5
    )

with scenario_col2:
    revenue_change = st.slider(
        "Revenue change %",
        min_value=-50,
        max_value=100,
        value=0,
        step=5
    )

with scenario_col3:
    risk_change = st.slider(
        "Risk change %",
        min_value=-50,
        max_value=100,
        value=0,
        step=5
    )

scenario_df = run_scenario_simulation(
    decision_df=decision_df,
    demand_change_pct=demand_change,
    revenue_change_pct=revenue_change,
    risk_change_pct=risk_change
)

scenario_display = scenario_df.copy()

scenario_display["scenario_forecast_demand_million"] = (
    scenario_display["scenario_forecast_demand"] / 1_000_000
).round(2)

scenario_display["scenario_forecast_revenue_crore"] = (
    scenario_display["scenario_forecast_revenue"] / 10_000_000
).round(2)

scenario_display["scenario_opportunity_score"] = (
    scenario_display["scenario_opportunity_score"]
).round(2)

scenario_display["scenario_risk_score"] = (
    scenario_display["scenario_risk_score"]
).round(2)

scenario_cols = [
    "scenario_rank",
    "state",
    "recommended_action",
    "scenario_action",
    "action_changed",
    "scenario_forecast_demand_million",
    "scenario_forecast_revenue_crore",
    "scenario_opportunity_score",
    "scenario_risk_score"
]

st.dataframe(
    scenario_display[scenario_cols],
    use_container_width=True,
    hide_index=True
)

changed_count = int((scenario_display["action_changed"] == "Changed").sum())

st.metric(
    "States with Changed Recommendation",
    changed_count
)

fig_scenario = px.bar(
    scenario_display.sort_values("scenario_opportunity_score", ascending=True),
    x="scenario_opportunity_score",
    y="state",
    color="scenario_action",
    orientation="h",
    title="Scenario-Based State Priority Ranking",
    labels={
        "scenario_opportunity_score": "Scenario Opportunity Score",
        "state": "State",
        "scenario_action": "Scenario Action"
    }
)

fig_scenario.update_layout(height=650)

st.plotly_chart(fig_scenario, use_container_width=True)


def run_scenario_simulation(
    decision_df,
    demand_change_pct=0,
    revenue_change_pct=0,
    risk_change_pct=0
):
    """
    Runs a what-if scenario on decision output.
    """

    df = decision_df.copy()

    df["scenario_forecast_demand"] = (
        df["forecast_demand"] * (1 + demand_change_pct / 100)
    )

    df["scenario_forecast_revenue"] = (
        df["forecast_revenue"] * (1 + revenue_change_pct / 100)
    )

    df["scenario_risk_score"] = (
        df["risk_score"] * (1 + risk_change_pct / 100)
    ).clip(0, 1)

    df["scenario_demand_score"] = normalize(df["scenario_forecast_demand"])
    df["scenario_revenue_score"] = normalize(df["scenario_forecast_revenue"])
    df["scenario_infrastructure_score"] = normalize(df["hospital_count"])
    df["scenario_growth_score"] = normalize(df["growth_rate"])

    df["scenario_opportunity_score"] = (
        0.35 * df["scenario_revenue_score"]
        + 0.30 * df["scenario_demand_score"]
        + 0.20 * df["scenario_infrastructure_score"]
        + 0.10 * df["scenario_growth_score"]
        - 0.15 * df["scenario_risk_score"]
    ).clip(0, 1)

    df["scenario_risk_adjusted_score"] = (
        df["scenario_opportunity_score"] * (1 - df["scenario_risk_score"])
    )

    def scenario_action(row):
        score = row["scenario_opportunity_score"]
        risk = row["scenario_risk_score"]

        if score >= 0.70 and risk <= 0.50:
            return "Invest Aggressively"

        elif score >= 0.65 and risk > 0.50:
            return "Pilot First"

        elif score >= 0.50:
            return "Selective Expansion"

        elif score >= 0.25:
            return "Maintain and Monitor"

        else:
            return "Deprioritize"

    df["scenario_action"] = df.apply(scenario_action, axis=1)

    df["action_changed"] = np.where(
        df["recommended_action"] != df["scenario_action"],
        "Changed",
        "No Change"
    )

    df["scenario_rank"] = df["scenario_opportunity_score"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    return df.sort_values("scenario_rank")


# -----------------------------
# Targeted state scenario simulation
# -----------------------------

st.markdown("---")
st.subheader("Targeted State Scenario Simulation")

st.markdown(
    """
    Use this section to test how the recommendation changes when only one selected state's assumptions change.
    """
)

target_state = st.selectbox(
    "Select state for targeted scenario",
    sorted(decision_df["state"].dropna().unique())
)

target_col1, target_col2, target_col3 = st.columns(3)

with target_col1:
    target_demand_change = st.slider(
        "Target state demand change %",
        min_value=-50,
        max_value=150,
        value=20,
        step=5
    )

with target_col2:
    target_revenue_change = st.slider(
        "Target state revenue change %",
        min_value=-50,
        max_value=150,
        value=20,
        step=5
    )

with target_col3:
    target_risk_change = st.slider(
        "Target state risk change %",
        min_value=-80,
        max_value=150,
        value=0,
        step=5
    )

target_scenario_df = run_targeted_state_scenario(
    decision_df=decision_df,
    target_state=target_state,
    demand_change_pct=target_demand_change,
    revenue_change_pct=target_revenue_change,
    risk_change_pct=target_risk_change
)

target_display = target_scenario_df.copy()

target_display["scenario_forecast_demand_million"] = (
    target_display["scenario_forecast_demand"] / 1_000_000
).round(2)

target_display["scenario_forecast_revenue_crore"] = (
    target_display["scenario_forecast_revenue"] / 10_000_000
).round(2)

target_display["scenario_opportunity_score"] = (
    target_display["scenario_opportunity_score"]
).round(2)

target_display["scenario_risk_score"] = (
    target_display["scenario_risk_score"]
).round(2)

target_display["rank_change"] = target_display["rank_change"].round(0).astype(int)

target_cols = [
    "scenario_rank",
    "state",
    "priority_rank",
    "rank_change",
    "recommended_action",
    "scenario_action",
    "action_changed",
    "scenario_forecast_demand_million",
    "scenario_forecast_revenue_crore",
    "scenario_opportunity_score",
    "scenario_risk_score"
]

st.dataframe(
    target_display[target_cols],
    use_container_width=True,
    hide_index=True
)

selected_row = target_display[target_display["state"] == target_state].iloc[0]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Original Rank",
        int(selected_row["priority_rank"])
    )

with c2:
    st.metric(
        "Scenario Rank",
        int(selected_row["scenario_rank"])
    )

with c3:
    st.metric(
        "Rank Change",
        int(selected_row["rank_change"])
    )

with c4:
    st.metric(
        "Action Changed?",
        selected_row["action_changed"]
    )

st.info(
    f"""
    **{target_state} scenario result**

    Original action: **{selected_row['recommended_action']}**

    Scenario action: **{selected_row['scenario_action']}**

    Original rank: **{int(selected_row['priority_rank'])}**

    Scenario rank: **{int(selected_row['scenario_rank'])}**
    """
)

fig_target = px.bar(
    target_display.sort_values("scenario_opportunity_score", ascending=True),
    x="scenario_opportunity_score",
    y="state",
    color="scenario_action",
    orientation="h",
    title=f"Targeted Scenario Ranking: {target_state}",
    labels={
        "scenario_opportunity_score": "Scenario Opportunity Score",
        "state": "State",
        "scenario_action": "Scenario Action"
    }
)

fig_target.update_layout(height=650)

st.plotly_chart(fig_target, use_container_width=True)


# -----------------------------
# Required improvement analysis
# -----------------------------

st.markdown("---")
st.subheader("Required Improvement Analysis")

st.markdown(
    """
    This section estimates how much demand/revenue improvement is required
    for the selected state to move into a better decision category.
    """
)

improvement_result = find_required_improvement_for_next_action(
    decision_df=decision_df,
    target_state=target_state,
    max_increase_pct=300,
    step_pct=5
)

if improvement_result["required_improvement_pct"] is not None:
    st.success(improvement_result["message"])
else:
    st.warning(improvement_result["message"])

imp_col1, imp_col2, imp_col3 = st.columns(3)

with imp_col1:
    st.metric(
        "Current Action",
        improvement_result["current_action"]
    )

with imp_col2:
    st.metric(
        "Target Action",
        improvement_result["target_action"]
    )

with imp_col3:
    required_value = improvement_result["required_improvement_pct"]

    if required_value is None:
        st.metric("Required Improvement", "Not reachable")
    else:
        st.metric("Required Improvement", f"+{required_value}%")


# -----------------------------
# Inventory decision section
# -----------------------------

if show_inventory:
    st.markdown("---")
    st.subheader("Inventory / Supply Decision")

    st.caption(
        f"""
        Current stock is assumed as **{stock_percentage}%** of forecast demand.
        Service level selected: **{service_level}**.
        """
    )

    inventory_cols = [
        "state",
        "forecast_demand_million",
        "current_stock_million",
        "safety_stock_million",
        "target_stock_million",
        "recommended_order_quantity_million",
        "inventory_action"
    ]

    available_inventory_cols = [
        col for col in inventory_cols
        if col in display_df.columns
    ]

    st.dataframe(
        display_df[available_inventory_cols],
        use_container_width=True,
        hide_index=True
    )

    fig3 = px.bar(
        decision_df.sort_values("recommended_order_quantity", ascending=True),
        x="recommended_order_quantity",
        y="state",
        color="inventory_action",
        orientation="h",
        title="Recommended Order Quantity by State",
        labels={
            "recommended_order_quantity": "Recommended Order Quantity",
            "state": "State",
            "inventory_action": "Inventory Action"
        }
    )

    fig3.update_layout(
        height=650,
        yaxis_title="State",
        xaxis_title="Recommended Order Quantity"
    )

    st.plotly_chart(fig3, use_container_width=True)


# -----------------------------
# Budget-based investment plan
# -----------------------------

st.markdown("---")
st.subheader("Budget-Based Investment Plan")

budget_df = decision_df.copy()

budget_df["estimated_investment_cost_cr"] = (
    budget_df["forecast_revenue"] * 0.02
) / 10_000_000

# Only invest in useful action categories
budget_df = budget_df[
    budget_df["recommended_action"].isin(
        [
            "Invest Aggressively",
            "Pilot First",
            "Selective Expansion"
        ]
    )
].copy()

budget_df = budget_df.sort_values("opportunity_score", ascending=False)

selected_states = []
used_budget = 0

for _, row in budget_df.iterrows():
    cost = row["estimated_investment_cost_cr"]

    if used_budget + cost <= available_budget:
        selected_states.append(
            {
                "state": row["state"],
                "recommended_action": row["recommended_action"],
                "opportunity_score": row["opportunity_score"],
                "estimated_investment_cost_cr": cost
            }
        )

        used_budget += cost

budget_col1, budget_col2, budget_col3 = st.columns(3)

with budget_col1:
    st.metric(
        "Available Budget",
        f"₹{available_budget:,.0f} Cr"
    )

with budget_col2:
    st.metric(
        "Used Budget",
        f"₹{used_budget:,.2f} Cr"
    )

with budget_col3:
    st.metric(
        "States Selected",
        len(selected_states)
    )

if selected_states:
    selected_budget_df = pd.DataFrame(selected_states)

    selected_budget_df["opportunity_score"] = (
        selected_budget_df["opportunity_score"].round(2)
    )

    selected_budget_df["estimated_investment_cost_cr"] = (
        selected_budget_df["estimated_investment_cost_cr"].round(2)
    )

    st.dataframe(
        selected_budget_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "No eligible state fits within the selected budget. Increase the budget or review investment criteria."
    )


# -----------------------------
# Advanced optimization section
# -----------------------------

st.markdown("---")
st.subheader("Advanced Optimized Investment Portfolio")

max_avg_risk = st.sidebar.slider(
    "Maximum average risk allowed",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

try:
    optimized_df = optimize_investment_portfolio(
        decision_df=decision_df,
        budget_cr=available_budget,
        max_avg_risk=max_avg_risk
    )

    if optimized_df.empty:
        st.warning(
            "No optimized portfolio found under the current budget and risk constraints."
        )
    else:
        optimized_display = optimized_df.copy()

        optimized_display["forecast_revenue_crore"] = (
            optimized_display["forecast_revenue"] / 10_000_000
        ).round(2)

        optimized_display["estimated_investment_cost_cr"] = (
            optimized_display["estimated_investment_cost_cr"]
        ).round(2)

        optimized_display["expected_value"] = (
            optimized_display["expected_value"]
        ).round(2)

        optimized_display["opportunity_score"] = (
            optimized_display["opportunity_score"]
        ).round(2)

        optimized_display["risk_score"] = (
            optimized_display["risk_score"]
        ).round(2)

        show_cols = [
            "state",
            "recommended_action",
            "forecast_revenue_crore",
            "estimated_investment_cost_cr",
            "expected_value",
            "opportunity_score",
            "risk_score"
        ]

        st.dataframe(
            optimized_display[show_cols],
            use_container_width=True,
            hide_index=True
        )

        total_cost = optimized_display["estimated_investment_cost_cr"].sum()
        total_expected_value = optimized_display["expected_value"].sum()
        avg_risk = optimized_display["risk_score"].mean()

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Optimized Cost", f"₹{total_cost:.2f} Cr")

        with c2:
            st.metric("Expected Value", f"₹{total_expected_value:.2f} Cr")

        with c3:
            st.metric("Average Risk", f"{avg_risk:.2f}")

except Exception as e:
    st.error("Optimization failed.")
    st.exception(e)    


# -----------------------------
# Top priority recommendations
# -----------------------------

st.markdown("---")
st.subheader("Top Priority Recommendations")

top_states = decision_df.sort_values("priority_rank").head(5)

for _, row in top_states.iterrows():
    st.markdown(
        f"""
        ### Rank {int(row['priority_rank'])}: {row['state']}

        **Recommended Action:** {row['recommended_action']}

        **Opportunity Score:** {row['opportunity_score']:.2f}

        **Risk Score:** {row['risk_score']:.2f}

        **Decision Confidence:** {row.get('decision_confidence', 'N/A')}

        **Reason:** {row['reason']}
        """
    )


# -----------------------------
# Final executive action plan
# -----------------------------

st.markdown("---")
st.subheader("Final Executive Action Plan")

invest_states = decision_df[
    decision_df["recommended_action"] == "Invest Aggressively"
]["state"].tolist()

selective_states = decision_df[
    decision_df["recommended_action"] == "Selective Expansion"
]["state"].tolist()

monitor_states = decision_df[
    decision_df["recommended_action"] == "Maintain and Monitor"
]["state"].tolist()

deprioritize_states = decision_df[
    decision_df["recommended_action"] == "Deprioritize"
]["state"].tolist()

st.markdown("### 1. Immediate Investment Priority")

if invest_states:
    st.success(
        "Prioritize aggressive investment in: "
        + ", ".join(invest_states)
    )
else:
    st.info("No states currently qualify for aggressive investment.")

st.markdown("### 2. Controlled Expansion Markets")

if selective_states:
    st.info(
        "Run selective expansion in: "
        + ", ".join(selective_states)
    )
else:
    st.info("No states currently qualify for selective expansion.")

st.markdown("### 3. Maintain and Monitor Markets")

if monitor_states:
    st.warning(
        "Maintain current strategy and monitor: "
        + ", ".join(monitor_states[:10])
        + ("..." if len(monitor_states) > 10 else "")
    )
else:
    st.info("No monitor-only states found.")

st.markdown("### 4. Deprioritized Markets")

if deprioritize_states:
    st.error(
        "Avoid major investment for now in: "
        + ", ".join(deprioritize_states[:10])
        + ("..." if len(deprioritize_states) > 10 else "")
    )
else:
    st.success("No states are currently deprioritized.")


# -----------------------------
# Save and download
# -----------------------------

decision_df.to_csv(save_path, index=False)

csv = decision_df.to_csv(index=False).encode("utf-8")

st.markdown("---")

st.download_button(
    label="Download Decision Recommendations CSV",
    data=csv,
    file_name="decision_recommendations.csv",
    mime="text/csv"
)

st.success(f"Decision recommendations saved to: {save_path}")