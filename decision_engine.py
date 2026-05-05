# decision_engine.py

import pandas as pd
import numpy as np


def normalize(series):
    """
    Normalize a numeric column between 0 and 1.
    """
    series = pd.to_numeric(series, errors="coerce").fillna(0)

    if series.max() == series.min():
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - series.min()) / (series.max() - series.min())


def find_column(df, possible_names):
    """
    Finds a matching column from many possible names.
    This makes the decision engine flexible with your existing CSV files.
    """
    cols_lower = {col.lower().strip(): col for col in df.columns}

    for name in possible_names:
        name = name.lower().strip()
        if name in cols_lower:
            return cols_lower[name]

    for col in df.columns:
        col_lower = col.lower().strip()
        for name in possible_names:
            if name.lower().strip() in col_lower:
                return col

    return None


def classify_action(row):
    """
    Converts score and risk into a business decision.
    """

    score = row["opportunity_score"]
    risk = row["risk_score"]

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


def decision_reason(row):
    """
    Creates a simple explanation for each recommendation.
    """

    action = row["recommended_action"]

    if action == "Invest Aggressively":
        return "High demand/revenue opportunity with acceptable risk. Prioritize supply, sales, and distribution."

    elif action == "Pilot First":
        return "Strong opportunity but uncertainty is high. Test with limited rollout before full investment."

    elif action == "Selective Expansion":
        return "Moderate opportunity. Expand only in selected regions or channels."

    elif action == "Maintain and Monitor":
        return "Stable market but not a top priority. Continue monitoring demand and risk."

    else:
        return "Low opportunity compared with other states. Avoid major investment for now."


def build_decision_table(
    forecast_df,
    hospital_df=None,
    bayesian_df=None
):
    """
    Converts your forecast outputs into a decision-making table.

    Uses:
    - forecast_combined_state_year.csv
    - hospitals_by_state.csv, optional
    - bayesian_state_intervals.csv, optional
    """

    df = forecast_df.copy()

    # Detect important columns automatically
    state_col = find_column(df, ["state", "state_name", "location"])
    year_col = find_column(df, ["year", "forecast_year"])

    demand_col = find_column(
        df,
        [
            "forecast_demand",
            "predicted_demand",
            "demand",
            "patients",
            "estimated_patients",
            "diabetes_patients",
            "cases"
        ]
    )

    revenue_col = find_column(
        df,
        [
            "forecast_revenue",
            "predicted_revenue",
            "revenue",
            "market_size",
            "sales",
            "value"
        ]
    )

    if state_col is None:
        raise ValueError("Could not find state column in forecast file.")

    if demand_col is None and revenue_col is None:
        raise ValueError("Could not find demand or revenue column in forecast file.")

    # If revenue is missing, use demand as proxy
    if revenue_col is None:
        revenue_col = demand_col
        df["estimated_revenue_proxy"] = df[demand_col]
        revenue_col = "estimated_revenue_proxy"

    # If demand is missing, use revenue as proxy
    if demand_col is None:
        demand_col = revenue_col
        df["estimated_demand_proxy"] = df[revenue_col]
        demand_col = "estimated_demand_proxy"

    # Keep latest forecast year
    if year_col is not None:
        latest_year = df[year_col].max()
        latest_df = df[df[year_col] == latest_year].copy()
    else:
        latest_df = df.copy()
        latest_df["decision_year"] = "Latest"
        year_col = "decision_year"

    # Merge hospital data if available
    if hospital_df is not None:
        hdf = hospital_df.copy()

        hospital_state_col = find_column(hdf, ["state", "state_name", "location"])
        hospital_count_col = find_column(
            hdf,
            [
                "hospitals",
                "hospital_count",
                "total_hospitals",
                "number_of_hospitals",
                "facilities"
            ]
        )

        if hospital_state_col is not None and hospital_count_col is not None:
            hdf = hdf[[hospital_state_col, hospital_count_col]].copy()
            hdf = hdf.rename(
                columns={
                    hospital_state_col: state_col,
                    hospital_count_col: "hospital_count"
                }
            )

            latest_df = latest_df.merge(hdf, on=state_col, how="left")
        else:
            latest_df["hospital_count"] = np.nan
    else:
        latest_df["hospital_count"] = np.nan

    # Merge Bayesian uncertainty if available
    if bayesian_df is not None:
        bdf = bayesian_df.copy()

        bayes_state_col = find_column(bdf, ["state", "state_name", "location"])

        lower_col = find_column(
            bdf,
            [
                "lower",
                "lower_bound",
                "lower_ci",
                "lower_interval",
                "credible_lower",
                "hdi_lower"
            ]
        )

        upper_col = find_column(
            bdf,
            [
                "upper",
                "upper_bound",
                "upper_ci",
                "upper_interval",
                "credible_upper",
                "hdi_upper"
            ]
        )

        if bayes_state_col is not None and lower_col is not None and upper_col is not None:
            bdf = bdf[[bayes_state_col, lower_col, upper_col]].copy()
            bdf = bdf.rename(
                columns={
                    bayes_state_col: state_col,
                    lower_col: "lower_interval",
                    upper_col: "upper_interval"
                }
            )

            latest_df = latest_df.merge(bdf, on=state_col, how="left")
        else:
            latest_df["lower_interval"] = np.nan
            latest_df["upper_interval"] = np.nan
    else:
        latest_df["lower_interval"] = np.nan
        latest_df["upper_interval"] = np.nan

    # Scores
    latest_df["demand_score"] = normalize(latest_df[demand_col])
    latest_df["revenue_score"] = normalize(latest_df[revenue_col])
    latest_df["infrastructure_score"] = normalize(latest_df["hospital_count"])

    # Growth score
    if year_col != "decision_year":
        first_year = df[year_col].min()

        base_df = df[df[year_col] == first_year][[state_col, revenue_col]].copy()
        base_df = base_df.rename(columns={revenue_col: "base_revenue"})

        latest_df = latest_df.merge(base_df, on=state_col, how="left")

        latest_df["growth_rate"] = (
            (latest_df[revenue_col] - latest_df["base_revenue"])
            / latest_df["base_revenue"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        latest_df["growth_score"] = normalize(latest_df["growth_rate"])
    else:
        latest_df["growth_rate"] = 0
        latest_df["growth_score"] = 0.5

    # Risk score from Bayesian interval if available
    interval_available = (
        "lower_interval" in latest_df.columns
        and "upper_interval" in latest_df.columns
        and latest_df["lower_interval"].notna().any()
        and latest_df["upper_interval"].notna().any()
    )

    if interval_available:
        latest_df["uncertainty_width"] = (
            latest_df["upper_interval"] - latest_df["lower_interval"]
        ) / latest_df[revenue_col].replace(0, np.nan)

        latest_df["uncertainty_width"] = (
            latest_df["uncertainty_width"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        latest_df["risk_score"] = latest_df["uncertainty_width"].rank(pct=True)
        latest_df["risk_score"] = latest_df["risk_score"].fillna(0.30)

    else:
        latest_df["uncertainty_width"] = np.nan
        latest_df["risk_score"] = 0.30

    # Decision confidence label based on risk score
    def confidence_label(risk):
        if risk <= 0.25:
            return "High Confidence"
        elif risk <= 0.60:
            return "Medium Confidence"
        else:
            return "Low Confidence"

    latest_df["decision_confidence"] = latest_df["risk_score"].apply(confidence_label)

    # Final opportunity score
    latest_df["opportunity_score"] = (
        0.35 * latest_df["revenue_score"]
        + 0.30 * latest_df["demand_score"]
        + 0.20 * latest_df["infrastructure_score"]
        + 0.10 * latest_df["growth_score"]
        - 0.15 * latest_df["risk_score"]
    )

    latest_df["opportunity_score"] = latest_df["opportunity_score"].clip(0, 1)
    latest_df["risk_adjusted_score"] = (
    latest_df["opportunity_score"] * (1 - latest_df["risk_score"])
    )

    # Business recommendation
    latest_df["recommended_action"] = latest_df.apply(classify_action, axis=1)
    latest_df["reason"] = latest_df.apply(decision_reason, axis=1)

    latest_df["priority_rank"] = latest_df["opportunity_score"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    # Rename important columns for dashboard clarity
    result = latest_df.copy()

    result = result.rename(
        columns={
            state_col: "state",
            year_col: "year",
            demand_col: "forecast_demand",
            revenue_col: "forecast_revenue"
        }
    )

    output_columns = [
        "priority_rank",
        "state",
        "year",
        "forecast_demand",
        "forecast_revenue",
        "growth_rate",
        "hospital_count",
        "opportunity_score",
        "risk_score",
        "decision_confidence",
        "recommended_action",
        "reason"
    ]

    for col in output_columns:
        if col not in result.columns:
            result[col] = np.nan

    result = result[output_columns].sort_values("priority_rank")

    return result


def add_inventory_decision(
    decision_df,
    current_stock=None,
    service_level_multiplier=1.65
):
    """
    Adds simple inventory decision logic.

    If current_stock is not given, assumes current stock is 70% of forecast demand.
    """

    df = decision_df.copy()

    if current_stock is None:
        df["current_stock"] = df["forecast_demand"] * 0.70
    else:
        df["current_stock"] = current_stock

    df["estimated_demand_std"] = df["forecast_demand"] * 0.20

    df["safety_stock"] = service_level_multiplier * df["estimated_demand_std"]

    df["target_stock"] = df["forecast_demand"] + df["safety_stock"]

    df["recommended_order_quantity"] = (
        df["target_stock"] - df["current_stock"]
    ).clip(lower=0)

    df["inventory_action"] = np.where(
        df["recommended_order_quantity"] > 0,
        "Reorder / Increase Supply",
        "No Reorder Needed"
    )

    return df

def optimize_investment_portfolio(
    decision_df,
    budget_cr=100,
    max_avg_risk=0.50,
    allowed_actions=None
):
    """
    Optimizes state investment selection under budget and risk constraints.

    Parameters:
    - decision_df: output of build_decision_table()
    - budget_cr: available budget in crore
    - max_avg_risk: maximum average risk allowed
    - allowed_actions: which decision categories are eligible

    Returns:
    - optimized selected states dataframe
    """

    try:
        import pulp
    except ImportError:
        raise ImportError("Please install PuLP using: pip install pulp")

    df = decision_df.copy()

    if allowed_actions is None:
        allowed_actions = [
            "Invest Aggressively",
            "Pilot First",
            "Selective Expansion"
        ]

    df = df[df["recommended_action"].isin(allowed_actions)].copy()

    if df.empty:
        return pd.DataFrame()

    df["estimated_investment_cost_cr"] = (
        df["forecast_revenue"] * 0.02
    ) / 10_000_000

    df["expected_value"] = (
        df["opportunity_score"] * df["forecast_revenue"]
    ) / 10_000_000

    states = df["state"].tolist()

    model = pulp.LpProblem(
        "Pharma_State_Investment_Optimization",
        pulp.LpMaximize
    )

    x = {
        state: pulp.LpVariable(f"invest_{state}", cat="Binary")
        for state in states
    }

    cost = {
        row["state"]: row["estimated_investment_cost_cr"]
        for _, row in df.iterrows()
    }

    value = {
        row["state"]: row["expected_value"]
        for _, row in df.iterrows()
    }

    risk = {
        row["state"]: row["risk_score"]
        for _, row in df.iterrows()
    }

    # Objective: maximize expected opportunity value
    model += pulp.lpSum(value[state] * x[state] for state in states)

    # Budget constraint
    model += pulp.lpSum(cost[state] * x[state] for state in states) <= budget_cr

    # Average risk constraint
    model += pulp.lpSum(risk[state] * x[state] for state in states) <= (
        max_avg_risk * pulp.lpSum(x[state] for state in states)
    )

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_states = [
        state for state in states
        if x[state].value() == 1
    ]

    result = df[df["state"].isin(selected_states)].copy()

    if result.empty:
        return result

    result["selected_by_optimizer"] = True

    result = result.sort_values(
        "opportunity_score",
        ascending=False
    )

    return result


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

    df["scenario_opportunity_score"] = (
        0.35 * df["scenario_revenue_score"]
        + 0.30 * df["scenario_demand_score"]
        + 0.20 * df["scenario_infrastructure_score"]
        + 0.10 * normalize(df["growth_rate"])
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

def run_targeted_state_scenario(
    decision_df,
    target_state,
    demand_change_pct=0,
    revenue_change_pct=0,
    risk_change_pct=0
):
    """
    Runs a what-if scenario for one selected state only.
    This is more useful than changing all states equally.
    """

    df = decision_df.copy()

    df["scenario_forecast_demand"] = df["forecast_demand"]
    df["scenario_forecast_revenue"] = df["forecast_revenue"]
    df["scenario_risk_score"] = df["risk_score"]

    mask = df["state"] == target_state

    df.loc[mask, "scenario_forecast_demand"] = (
        df.loc[mask, "forecast_demand"] * (1 + demand_change_pct / 100)
    )

    df.loc[mask, "scenario_forecast_revenue"] = (
        df.loc[mask, "forecast_revenue"] * (1 + revenue_change_pct / 100)
    )

    df.loc[mask, "scenario_risk_score"] = (
        df.loc[mask, "risk_score"] * (1 + risk_change_pct / 100)
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

    df["rank_change"] = df["priority_rank"] - df["scenario_rank"]

    return df.sort_values("scenario_rank")


def find_required_improvement_for_next_action(
    decision_df,
    target_state,
    max_increase_pct=300,
    step_pct=5
):
    """
    Finds how much demand and revenue improvement is needed
    for a selected state to move to a better decision category.
    """

    df = decision_df.copy()

    current_row = df[df["state"] == target_state]

    if current_row.empty:
        return {
            "state": target_state,
            "current_action": None,
            "target_action": None,
            "required_improvement_pct": None,
            "message": "State not found."
        }

    current_action = current_row.iloc[0]["recommended_action"]

    action_order = {
        "Deprioritize": 1,
        "Maintain and Monitor": 2,
        "Selective Expansion": 3,
        "Pilot First": 4,
        "Invest Aggressively": 5
    }

    reverse_action_order = {
        1: "Maintain and Monitor",
        2: "Selective Expansion",
        3: "Invest Aggressively",
        4: "Invest Aggressively",
        5: "Already top category"
    }

    current_level = action_order.get(current_action, 1)

    if current_level >= 5:
        return {
            "state": target_state,
            "current_action": current_action,
            "target_action": "Already top category",
            "required_improvement_pct": 0,
            "message": f"{target_state} is already in the top decision category."
        }

    target_action = reverse_action_order[current_level]

    for increase in range(0, max_increase_pct + step_pct, step_pct):
        scenario_df = run_targeted_state_scenario(
            decision_df=df,
            target_state=target_state,
            demand_change_pct=increase,
            revenue_change_pct=increase,
            risk_change_pct=0
        )

        scenario_row = scenario_df[scenario_df["state"] == target_state].iloc[0]
        scenario_action = scenario_row["scenario_action"]

        if action_order.get(scenario_action, 1) > current_level:
            return {
                "state": target_state,
                "current_action": current_action,
                "target_action": scenario_action,
                "required_improvement_pct": increase,
                "message": (
                    f"{target_state} needs approximately +{increase}% demand/revenue improvement "
                    f"to move from {current_action} to {scenario_action}."
                )
            }

    return {
        "state": target_state,
        "current_action": current_action,
        "target_action": target_action,
        "required_improvement_pct": None,
        "message": (
            f"{target_state} did not move to a better category even after "
            f"+{max_increase_pct}% demand/revenue improvement."
        )
    }


def generate_decision_alerts(decision_df):
    """
    Generates management alerts from decision table.
    """

    alerts = []

    df = decision_df.copy()

    for _, row in df.iterrows():
        state = row["state"]

        if row["recommended_action"] == "Invest Aggressively":
            alerts.append(
                {
                    "severity": "High",
                    "state": state,
                    "alert_type": "Investment Priority",
                    "message": f"{state} is a high-priority investment market."
                }
            )

        if row["risk_score"] >= 0.70:
            alerts.append(
                {
                    "severity": "High",
                    "state": state,
                    "alert_type": "High Forecast Risk",
                    "message": f"{state} has high forecast uncertainty. Consider pilot before full rollout."
                }
            )

        if (
            "recommended_order_quantity" in df.columns
            and row.get("recommended_order_quantity", 0) > row["forecast_demand"] * 0.50
        ):
            alerts.append(
                {
                    "severity": "Medium",
                    "state": state,
                    "alert_type": "Large Supply Requirement",
                    "message": f"{state} requires a large inventory replenishment."
                }
            )

        if (
            row["recommended_action"] == "Selective Expansion"
            and row["risk_score"] <= 0.30
        ):
            alerts.append(
                {
                    "severity": "Medium",
                    "state": state,
                    "alert_type": "Expansion Candidate",
                    "message": f"{state} is a good candidate for controlled expansion."
                }
            )

    return pd.DataFrame(alerts)


def add_abc_classification(decision_df):
    """
    Adds ABC classification based on forecast revenue.
    """

    df = decision_df.copy()

    df = df.sort_values("forecast_revenue", ascending=False)

    total_revenue = df["forecast_revenue"].sum()

    if total_revenue == 0:
        df["revenue_share"] = 0
        df["cumulative_revenue_share"] = 0
        df["abc_class"] = "C"
        return df

    df["revenue_share"] = df["forecast_revenue"] / total_revenue
    df["cumulative_revenue_share"] = df["revenue_share"].cumsum()

    conditions = [
        df["cumulative_revenue_share"] <= 0.80,
        df["cumulative_revenue_share"] <= 0.95
    ]

    choices = ["A", "B"]

    df["abc_class"] = np.select(
        conditions,
        choices,
        default="C"
    )

    return df