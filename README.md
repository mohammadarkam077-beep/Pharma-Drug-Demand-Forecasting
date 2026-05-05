<div align="center">

# 💊 AI-Powered Pharma Demand Forecasting & Decision Optimization System

### Forecast demand. Quantify risk. Optimize investment. Plan inventory. Support executive decisions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple?style=for-the-badge&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly)
![Optimization](https://img.shields.io/badge/Optimization-PuLP-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

</div>

---

## 📌 Project Summary

This project is an **end-to-end pharmaceutical demand forecasting and decision optimization platform** built for strategic pharma market planning.

It goes beyond traditional forecasting by converting predicted demand into **actionable business decisions** such as:

- ✅ Which states should receive aggressive investment?
- ✅ Which markets should be expanded selectively?
- ✅ Which states should be monitored or deprioritized?
- ✅ How much inventory should be prepared?
- ✅ Which states should be selected under a limited budget?
- ✅ What happens if demand, revenue, or risk assumptions change?
- ✅ How much improvement is needed for a state to move into a better decision category?

The final system works as a **Prescriptive Decision Support System** for pharmaceutical demand, supply, and investment planning.

---

## 🚀 Live System Capabilities

| Capability | Description |
|---|---|
| 📈 Demand Forecasting | Forecasts national and state-level pharmaceutical demand |
| 🗺️ State-Level Analysis | Ranks Indian states by demand, revenue, risk, and opportunity |
| 🔬 Bayesian Risk Estimation | Converts forecast uncertainty into business risk scores |
| 🎯 Sensitivity Analysis | Tests how input assumptions affect forecast outcomes |
| 🧭 Decision Engine | Converts forecasts into recommended business actions |
| 📦 Inventory Planning | Calculates safety stock, target stock, and reorder quantity |
| 💰 Budget Planning | Allocates limited investment budget across eligible states |
| 🧠 Portfolio Optimization | Uses optimization to select best investment portfolio |
| 🔁 Scenario Simulation | Tests what-if changes in demand, revenue, and risk |
| 🎯 Targeted State Simulation | Tests scenario impact on a selected state |
| 📊 Opportunity-Risk Matrix | Visualizes states by opportunity and risk |
| 🚨 Management Alerts | Generates executive-level decision alerts |
| 🧾 Final Action Plan | Produces management-ready recommendations |

---

## 🧠 Problem Statement

Pharmaceutical companies must make high-impact decisions under uncertainty:

- Where should they invest?
- Which regions need more inventory?
- Which markets are risky?
- Which states have high growth potential?
- How should limited budget be allocated?

Traditional forecasting systems only answer:

```text
What will demand be?

🏗️ System Architecture
text
                    ┌──────────────────────────┐
                    │      Raw Health Data      │
                    │  Population, Diabetes,    │
                    │  Hospitals, Prevalence    │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │    Data Processing        │
                    │ Cleaning + Integration    │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   Forecasting Layer       │
                    │ National + State Forecast │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Bayesian Uncertainty      │
                    │ Risk + Confidence Scores  │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │     Decision Engine       │
                    │ Opportunity + Risk Score  │
                    └─────────────┬────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ Inventory Logic  │ │ Budget Optimizer │ │ Scenario Engine  │
   └──────────────────┘ └──────────────────┘ └──────────────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │ Streamlit Decision UI     │
                    │ Executive Dashboard       │
                    └──────────────────────────┘


🧭 Decision Intelligence Workflow
text
Forecast Demand
      ↓
Estimate Revenue
      ↓
Measure Forecast Risk
      ↓
Compute Opportunity Score
      ↓
Assign Decision Confidence
      ↓
Classify Business Action
      ↓
Generate Inventory Decision
      ↓
Run Scenario Simulation
      ↓
Optimize Budget Allocation
      ↓
Produce Executive Action Plan

📊 Dashboard Preview

🧭 Decision-Making Dashboard
markdown
![Decision Dashboard](screenshots/decision_dashboard.png)
📊 Opportunity-Risk Matrix
markdown
![Opportunity Risk Matrix](screenshots/opportunity_risk_matrix.png)
🔁 Scenario Simulation
markdown
![Scenario Simulation](screenshots/scenario_simulation.png)
📦 Inventory Decision
markdown
![Inventory Decision](screenshots/inventory_decision.png)
💰 Optimization Output
markdown
![Optimization Output](screenshots/optimization_output.png)

🧩 Dashboard Pages
Page	Purpose
📈 National Forecast	Shows national demand and revenue forecast
🗺️ State Explorer	Explores state-wise demand and opportunity
🔬 Bayesian	Displays uncertainty and forecast intervals
🎯 Sensitivity	Shows impact of key business assumptions
📋 Executive Summary	Summarizes high-level market opportunity
🧭 Decision Making	Converts forecasts into decisions and action plans

🔥 Core Features
1. 📈 National Demand Forecasting
The system forecasts national-level pharmaceutical demand and revenue.

Outputs:

Forecasted patient demand
Forecasted revenue
Demand trend plots
Revenue trend plots
Confidence visualization
2. 🗺️ State-Level Forecasting
State-level forecasting helps identify high-potential regional markets.

The system generates:

State-wise forecast demand
State-wise forecast revenue
State priority ranking
Hospital infrastructure integration
3. 🔬 Bayesian Uncertainty & Risk Scoring
Forecasts are uncertain. This system converts uncertainty into a business-friendly risk_score.

text
Low Risk    → High Confidence
Medium Risk → Medium Confidence
High Risk   → Low Confidence
This prevents decision-makers from investing blindly in high-demand but uncertain markets.

4. 🧭 Decision Engine
The decision engine converts forecast data into recommended business actions.

It considers:

Demand potential
Revenue potential
Growth rate
Hospital infrastructure
Forecast uncertainty
Risk score
Opportunity score
🧮 Opportunity Score Formula
text
Opportunity Score =
    0.35 × Revenue Score
  + 0.30 × Demand Score
  + 0.20 × Infrastructure Score
  + 0.10 × Growth Score
  - 0.15 × Risk Score

🏷️ Recommendation Categories
Condition	Recommendation
High opportunity + low risk	Invest Aggressively
High opportunity + high risk	Pilot First
Medium opportunity	Selective Expansion
Low-medium opportunity	Maintain and Monitor
Low opportunity	Deprioritize
✅ Example Decision Output
text
State: Maharashtra
Forecast Demand: 782.76M
Forecast Revenue: ₹1616.98 Cr
Opportunity Score: 0.89
Risk Score: 0.06
Decision Confidence: High Confidence
Recommended Action: Invest Aggressively
Reason: High demand/revenue opportunity with acceptable risk.
📊 Opportunity-Risk Matrix
The system includes a strategic 2x2 decision matrix.

Quadrant	Meaning	Action
High Opportunity + Low Risk	Best markets	Scale / Invest
High Opportunity + High Risk	Promising but uncertain	Pilot / Validate
Low Opportunity + Low Risk	Stable but limited	Maintain
Low Opportunity + High Risk	Weak and risky	Avoid / Deprioritize
This matrix gives executives a quick visual map of where to invest, pilot, maintain, or avoid.

🚨 Management Alerts
The dashboard automatically generates executive alerts.

Example alerts:

text
Maharashtra is a high-priority investment market.
Tamil Nadu is a good candidate for controlled expansion.
Maharashtra requires a large inventory replenishment.
Alert severity:

Severity	Meaning
High	Immediate executive attention
Medium	Planning or monitoring required
Low	Informational
📦 Inventory / Supply Planning
The system recommends inventory actions using:

Forecast demand
Current stock assumption
Safety stock
Target stock
Service level
Inventory Logic
text
Current Stock = Forecast Demand × Current Stock %
Safety Stock = Service Level Multiplier × Estimated Demand Standard Deviation
Target Stock = Forecast Demand + Safety Stock
Recommended Order Quantity = Target Stock - Current Stock
Service Levels
Service Level	Multiplier
90%	1.28
95%	1.65
98%	2.05
Inventory action output:

text
Reorder / Increase Supply
No Reorder Needed
💰 Budget-Based Investment Planning
Users can enter an available investment budget.

The system selects eligible states based on:

Opportunity score
Recommended action
Estimated investment cost
Available budget
Example:

text
Available Budget: ₹100 Cr
Used Budget: ₹82.68 Cr
Selected States: 3
🧠 Advanced Investment Optimization
The project includes portfolio optimization using PuLP.

Objective
text
Maximize total expected opportunity value
Constraints
text
Total investment cost ≤ Available budget
Average risk ≤ Maximum allowed risk
Only eligible states are selected
This makes the system more advanced than simple ranking.

🔁 Scenario Simulation
Users can test how decisions change when assumptions change.

Scenario controls:

Demand change %
Revenue change %
Risk change %
The system recalculates:

Scenario demand
Scenario revenue
Scenario opportunity score
Scenario recommendation
Recommendation changes
🎯 Targeted State Scenario
Users can select one state and test targeted improvements.

Example:

text
Target State: Kerala
Demand Change: +35%
Revenue Change: +35%
Risk Change: -10%
The system shows:

Original rank
Scenario rank
Rank change
Original action
Scenario action
📈 Required Improvement Analysis
This feature estimates how much improvement is required for a state to move into a better decision category.

Example:

text
Kerala needs approximately +35% demand/revenue improvement
to move from Maintain and Monitor to Selective Expansion.
This is useful for:

Market development planning
Sales strategy
Diagnosis campaign planning
Channel expansion decisions
🧾 ABC Inventory Classification
ABC classification groups states by revenue contribution.

Class	Meaning	Suggested Strategy
A	Highest revenue contribution	Highest service priority
B	Medium contribution	Standard monitoring
C	Lower contribution	Controlled inventory exposure
This supports smarter inventory and supply-chain planning.

📋 Final Executive Action Plan
The dashboard automatically creates a final management action plan:

text
1. Immediate Investment Priority
2. Controlled Expansion Markets
3. Maintain and Monitor Markets
4. Deprioritized Markets
This turns analysis into boardroom-ready recommendations.

🗂️ Project Structure
text
Pharma-Drug-Demand-Forecasting/
│
├── app.py
├── forecasting.py
├── state_forecasting.py
├── bayesian_forecasting.py
├── sensitivity_analysis.py
├── decision_engine.py
├── run_all.py
├── live_data_pipeline.py
├── process_real_data.py
├── process_state_data.py
│
├── data/
│   ├── hospitals_by_state.csv
│   ├── state_master.csv
│   ├── india_diabetes_real_master.csv
│   ├── state_prevalence_icmr.csv
│   ├── live_wb_population.csv
│   ├── live_wb_health_exp.csv
│   ├── live_who_diabetes.csv
│   └── ...
│
├── outputs/
│   ├── forecast_output.csv
│   ├── forecast_output_advanced.csv
│   ├── forecast_combined_state_year.csv
│   ├── bayesian_state_intervals.csv
│   ├── sensitivity_analysis.csv
│   ├── decision_recommendations.csv
│   ├── demand_plot.png
│   ├── revenue_plot.png
│   ├── tornado_chart.png
│   └── ...
│
├── pages/
│   ├── 1_📈_National_Forecast.py
│   ├── 2_🗺️_State_Explorer.py
│   ├── 3_🔬_Bayesian.py
│   ├── 4_🎯_Sensitivity.py
│   ├── 5_📋_Executive_Summary.py
│   └── 6_🧭_Decision_Making.py
│
├── requirements.txt
├── runtime.txt
└── README.md
🧠 Important Files
File	Purpose
app.py	Main Streamlit app entry point
forecasting.py	National demand and revenue forecasting
state_forecasting.py	State-level forecasting
bayesian_forecasting.py	Bayesian uncertainty estimation
sensitivity_analysis.py	Sensitivity and tornado analysis
decision_engine.py	Core decision intelligence logic
run_all.py	Runs the full pipeline
6_🧭_Decision_Making.py	Final decision dashboard
⚙️ Installation
1. Clone the repository
bash
git clone https://github.com/your-username/Pharma-Drug-Demand-Forecasting.git
cd Pharma-Drug-Demand-Forecasting
2. Create a virtual environment
bash
python -m venv venv
Activate it.

For Windows:

bash
venv\Scripts\activate
For macOS/Linux:

bash
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
If PuLP is missing:

bash
pip install pulp
📦 Requirements
text
streamlit
pandas
numpy
plotly
matplotlib
seaborn
scikit-learn
pulp
openpyxl
▶️ How to Run
Run full pipeline
bash
python run_all.py
Launch dashboard
bash
streamlit run app.py
📤 Generated Outputs
Output File	Description
forecast_output.csv	National forecast
forecast_output_advanced.csv	Advanced forecast output
forecast_combined_state_year.csv	State-year forecast
bayesian_state_intervals.csv	Bayesian intervals
sensitivity_analysis.csv	Sensitivity results
decision_recommendations.csv	Final decision recommendations
demand_plot.png	Demand visualization
revenue_plot.png	Revenue visualization
tornado_chart.png	Sensitivity tornado chart
🧾 Final Decision Output Columns
Column	Description
priority_rank	State priority rank
state	State name
year	Forecast year
forecast_demand	Forecasted demand
forecast_revenue	Forecasted revenue
growth_rate	Growth estimate
hospital_count	Hospital infrastructure
opportunity_score	Business opportunity score
risk_score	Forecast uncertainty score
decision_confidence	Confidence label
recommended_action	Final recommended action
reason	Business explanation
🏢 Business Impact
This system can help pharma companies with:

Strategic Planning
State prioritization
Market expansion planning
Launch strategy
Regional investment decisions
Supply Chain Planning
Inventory allocation
Safety stock planning
Stockout prevention
Service-level planning
Commercial Planning
Sales force allocation
Revenue opportunity assessment
Channel strategy
Market access planning
Risk Management
Forecast uncertainty monitoring
Pilot-vs-scale decision-making
Scenario testing
Confidence-based recommendations
📊 Analytics Maturity
Analytics Type	Implemented
Descriptive Analytics	✅
Diagnostic Analytics	✅
Predictive Analytics	✅
Prescriptive Analytics	✅
Scenario Analytics	✅
Optimization	✅
The system moves from:

text
What happened?
to:

text
What will happen?
to:

text
What should we do?
🧪 Example Use Case
A decision-maker selects Kerala in the targeted scenario module.

The system finds:

text
Current Action: Maintain and Monitor
Target Action: Selective Expansion
Required Improvement: +35%
Business interpretation:

text
Kerala requires approximately 35% improvement in demand/revenue
before it becomes eligible for selective expansion.
This insight can support:

Marketing investment
Diagnosis campaigns
Distribution expansion
Hospital partnerships
🛠️ Technology Stack
Layer	Tools
Programming	Python
Dashboard	Streamlit
Data Processing	Pandas, NumPy
Visualization	Plotly, Matplotlib
Forecasting	Python forecasting pipeline
Optimization	PuLP
Uncertainty	Bayesian-style intervals
Output	CSV, PNG
UI	Multi-page Streamlit app
🚀 Future Enhancements
Planned future improvements:

Product/SKU-level forecasting
District-level demand forecasting
ERP inventory integration
Supplier lead-time modeling
Stockout probability estimation
Multi-objective optimization
Automated retraining pipeline
API backend for decision recommendations
Role-based user access
Docker deployment
Cloud deployment on AWS/Azure/GCP
📌 Current Status
text
✅ Data pipeline
✅ National forecasting
✅ State forecasting
✅ Bayesian risk estimation
✅ Sensitivity analysis
✅ Decision engine
✅ Inventory planning
✅ Scenario simulation
✅ Targeted scenario simulation
✅ Required improvement analysis
✅ Budget planning
✅ Portfolio optimization
✅ Opportunity-risk matrix
✅ Management alerts
✅ Executive action plan
