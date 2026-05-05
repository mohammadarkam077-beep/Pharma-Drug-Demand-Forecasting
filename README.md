<div align="center">

<!-- HEADER BANNER -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0f4c81,100:00b4d8&height=200&section=header&text=Pharma%20Drug%20Demand%20Forecasting&fontSize=34&fontColor=ffffff&fontAlignY=38&desc=Patient-Based%20%7C%20Scenario%20Analysis%20%7C%20Monte%20Carlo%20Simulation&descAlignY=58&descSize=16&animation=fadeIn" />

<br/>

<!-- BADGES -->
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=plotly&logoColor=white)

<br/>

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Pharmaceutical%20Analytics-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Author](https://img.shields.io/badge/Author-Md%20Arkam-ff69b4?style=flat-square)

<br/><br/>

> **A production-ready demand forecasting engine for pharmaceutical markets — built at the intersection of pharmacoeconomics and data science.**

</div>

---

## 🧭 Table of Contents

| Section | Description |
|---|---|
| [📌 Overview](#-overview) | What this project does |
| [🎯 Objectives](#-objectives) | What we forecast |
| [🧠 Methodology](#-methodology) | The science behind the model |
| [📊 Key Results](#-key-results) | Numbers that matter |
| [💡 Business Insights](#-business-insights) | Strategic takeaways |
| [🛠️ Tech Stack](#️-tech-stack) | Tools used |
| [📁 Project Structure](#-project-structure) | Directory layout |
| [🚀 Quick Start](#-quick-start) | Get running in minutes |
| [📈 Outputs](#-outputs) | What you get |
| [🔮 Roadmap](#-roadmap) | What's coming next |

---

## 📌 Overview

This project builds a **patient-based pharmaceutical drug demand forecasting model** grounded in real-world epidemiological data and pharmacoeconomic assumptions.

It estimates **drug demand, revenue potential, and market growth trends** for a chronic disease indication (e.g., **Type 2 Diabetes**) in the Indian market — combining deterministic projections with probabilistic simulation techniques.

<div align="center">

```
Population Data  →  Epidemiology  →  Patient Pool  →  Demand  →  Revenue  →  Business Strategy
```

</div>

---

## 🎯 Objectives

<table>
<tr>
<td align="center" width="200">📦<br/><b>Drug Demand</b><br/><sub>Annual unit forecast</sub></td>
<td align="center" width="200">💰<br/><b>Revenue Potential</b><br/><sub>Market value in ₹ Cr</sub></td>
<td align="center" width="200">📈<br/><b>Growth Trends</b><br/><sub>CAGR & YoY analysis</sub></td>
<td align="center" width="200">⚠️<br/><b>Risk Modeling</b><br/><sub>Uncertainty via simulation</sub></td>
</tr>
</table>

---

## 🧠 Methodology

The forecasting pipeline follows a structured **5-step patient-based approach**:

```
┌─────────────────────────────────────────────────────────────────┐
│                   FORECASTING PIPELINE                          │
│                                                                 │
│  [1] Population Data                                            │
│       └─ Year-wise Indian population dataset                    │
│                    │                                            │
│                    ▼                                            │
│  [2] Epidemiological Parameters                                 │
│       ├─ Prevalence rate (time-varying)                         │
│       ├─ Diagnosis rate                                         │
│       ├─ Treatment rate                                         │
│       ├─ Market share (growth-adjusted)                         │
│       └─ Patient compliance                                     │
│                    │                                            │
│                    ▼                                            │
│  [3] Patient Pool Calculation                                   │
│       └─ Eligible patients = f(population × epi params)         │
│                    │                                            │
│                    ▼                                            │
│  [4] Demand & Revenue Estimation                                │
│       ├─ Demand = Patients × Daily Dose × 365                   │
│       └─ Revenue = Demand × Price per unit                      │
│                    │                                            │
│                    ▼                                            │
│  [5] Advanced Analytics                                         │
│       ├─ Scenario Analysis  (Worst / Base / Best)               │
│       ├─ Monte Carlo Simulation  (10,000 iterations)            │
│       ├─ Sensitivity Analysis                                   │
│       └─ CAGR & Peak Demand Identification                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

<div align="center">

| Metric | Value | Interpretation |
|:---|:---:|:---|
| 📦 **Demand Start (Base Year)** | 0.91 Billion Units | Baseline market size |
| 📦 **Demand End (Forecast Year)** | 3.13 Billion Units | Projected market size |
| 💰 **Revenue Start** | ₹10.8 Billion | Initial revenue potential |
| 💰 **Revenue End** | ₹37.5 Billion | Future revenue potential |
| 📈 **Market CAGR** | ~9.27% | Strong consistent growth |
| 🏔️ **Peak Demand Year** | 2024 | Inflection point identified |
| ↗️ **Total Demand Growth** | +2.23 Billion Units | Absolute market expansion |
| 💎 **Revenue Potential** | ₹3,700+ Cr | Investable opportunity size |

</div>

---

## 💡 Business Insights

<details>
<summary><b>📌 Click to expand full strategic analysis</b></summary>

<br/>

### ✅ What the Model Reveals

**1. Growth Is Structural, Not Cyclical**
The demand growth is driven by expanding patient pools — a result of rising prevalence, improving diagnosis infrastructure, and treatment access — making this a durable, long-term trend rather than a short-term spike.

**2. Treatment Penetration Is the Leverage Point**
Even small improvements in diagnosis and treatment rates create outsized demand growth. This makes awareness campaigns and HCP engagement highly ROI-positive.

**3. Market CAGR Outpaces GDP Growth**
At ~9.27%, the market CAGR signals pharmaceutical companies should be **building capacity now**, not reactively.

**4. Risk Is Manageable Under Monte Carlo**
The simulation shows demand remains positive across 95%+ of scenarios, suggesting robust fundamentals even under pessimistic assumptions.

### ⚠️ Risks & Limitations

- Pricing assumptions are **static** — real-world pricing dynamics (generics, tenders, NPPA caps) could compress margins
- Regional variation not captured — urban vs. rural treatment gaps may skew national-level insights
- Competition from biosimilars / new entrants not modeled

</details>

---

## 🛠️ Tech Stack

<div align="center">

| Tool | Role | Version |
|:---:|:---:|:---:|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat-square) | Core language | 3.8+ |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat-square) | Data manipulation | Latest |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat-square) | Monte Carlo simulation | Latest |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square) | Visualization | Latest |

</div>

---

## 📁 Project Structure

```
📦 Pharma-Drug-Demand-Forecasting/
│
├── 🐍 forecasting.py              # Main forecasting engine
│
├── 📂 data/
│   └── 📄 India_dataset_population.csv   # Population dataset (India)
│
├── 📂 outputs/
│   ├── 📊 demand_plot.png         # Demand forecast visualization
│   ├── 💰 revenue_plot.png        # Revenue forecast visualization
│   └── 📁 forecast_output.csv    # Full forecast results (tabular)
│
├── 📋 requirements.txt            # Python dependencies
└── 📖 README.md                   # You are here
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/mohammadarkam077-beep/Pharma-Drug-Demand-Forecasting.git

# 2. Navigate into the project
cd Pharma-Drug-Demand-Forecasting

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the forecasting model
python forecasting.py
```

> ✅ Outputs will be saved automatically to the `/outputs` folder.

---

## 📈 Outputs

After running `forecasting.py`, you'll find:

| Output File | Type | Description |
|---|---|---|
| `demand_plot.png` | 📊 Chart | Year-wise drug demand forecast (Worst/Base/Best scenarios) |
| `revenue_plot.png` | 💰 Chart | Revenue projection with Monte Carlo confidence intervals |
| `forecast_output.csv` | 📁 Data | Complete numerical forecast table for further analysis |

---

## 🔮 Roadmap

- [ ] 🌐 Integrate real-world datasets (WHO, IHME, NFHS)
- [ ] 🗺️ Add regional segmentation (State-level forecasting)
- [ ] 📊 Build an interactive dashboard (Power BI / Streamlit)
- [ ] 💲 Model dynamic pricing (generic competition, NPPA regulations)
- [ ] 🏥 Expand to multiple therapeutic areas
- [ ] 🤖 ML-based prevalence rate prediction

---

## 👤 Author

<div align="center">

**Md Arkam**
*MPharm · Pharmaceutical Sciences + Data Analytics*

[![GitHub](https://img.shields.io/badge/GitHub-mohammadarkam077--beep-181717?style=for-the-badge&logo=github)](https://github.com/mohammadarkam077-beep)

</div>

---

<div align="center">

**If this project helped you, give it a ⭐ — it means a lot!**

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00b4d8,100:0f4c81&height=100&section=footer" />

</div>
