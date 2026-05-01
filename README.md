<div align="center">

# 💊 India Pharma Forecasting Platform

### 🧠 Strategic Intelligence for India's Diabetes Pharmaceutical Market

*A production-grade Bayesian forecasting platform analyzing diabetes drug demand across 31+ Indian states using real-world epidemiological data from ICMR, WHO, IHME, NFHS-5 & World Bank.*

<br>

[![Live App](https://img.shields.io/badge/🚀_Live_Demo-pharma--forecasting.streamlit.app-FF4B4B?style=for-the-badge)](https://pharma-forecasting.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

### 🔗 [**Try the Live Application →**](https://pharma-forecasting.streamlit.app/)

---

</div>

## 🎯 Executive Summary

A full-stack analytics platform that fuses **real epidemiological data** (ICMR, WHO Global Health Observatory, IHME GBD 2023, NFHS-5, World Bank) with **rigorous hierarchical Bayesian modeling** to forecast diabetes drug demand across India's 31+ states.

> 💡 **Key Discovery:** **₹24,466 Cr untapped market** (280% larger than current ₹8,751 Cr) — only **31% of India's 81.5M diabetics are currently diagnosed**. Top 5 states represent **46% of total revenue**.

This isn't a toy project. It's a complete data pipeline, statistical model, and decision-support dashboard built to answer the question: **"Where should pharma companies invest next, and how confident can they be in those forecasts?"**

---

## 📊 Headline Metrics

<div align="center">

| 💰 Market Size | 👥 Diabetic Population | 🗺️ States Modeled | 📈 5Y Growth | 🔓 Untapped |
|:---:|:---:|:---:|:---:|:---:|
| **₹8,751 Cr** | **81.5 M** | **31+** | **+96%** | **₹24,466 Cr** |

</div>

---

## ✨ Core Modules

### 🏠 **Home Dashboard** (`app.py`)
Landing page with KPIs, key insights, and navigation to all analytical modules.

### 📊 **National Forecast** (`pages/1_📈_National_Forecast.py`)
- 5-year demand forecasts at the country level
- Confidence intervals using Bayesian credible bounds
- Patient funnel visualization
- Revenue projections

### 🗺️ **State Explorer** (`pages/2_📖_State_Explorer.py`)
- Interactive state-level drill-downs across 31+ states
- Hospital infrastructure mapping (`hospital_directory.csv`)
- ICMR state-level prevalence integration
- Comparative state analytics

### 🔬 **Bayesian Uncertainty Quantification** (`pages/3_🔬_Bayesian.py`)
- **Hierarchical 3-level model**: National → Regional → State
- **Sampler**: nutpie (Rust-based NUTS) — 4 chains × 1,000 draws + 500 tune
- **Reparameterization**: Non-centered + logit-scale for sampler stability
- Posterior distribution plots & state-level credible intervals

### 🎯 **Sensitivity Analysis** (`pages/4_🎯_Sensitivity.py`)
- Tornado charts ranking parameter influence
- Scenario comparison (optimistic / baseline / pessimistic)
- One-at-a-time (OAT) sensitivity scoring

### 📋 **Executive Summary** (`pages/5_📋_Executive_Summary.py`)
- C-suite ready strategic insights
- Market opportunity quantification
- Investment recommendations

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|:---|:---|
| 🎨 **Frontend & UI** | Streamlit 1.39.0, custom theme (`.streamlit/config.toml`) |
| 🐍 **Core Language** | Python 3.11 |
| 📊 **Data Processing** | Pandas 2.2.3, NumPy 1.26.4, OpenPyXL 3.1.5 |
| 📈 **Visualization** | Plotly 5.24.1, Pillow 10.4.0 |
| 🎲 **Bayesian Inference** | nutpie (Rust NUTS sampler), PyMC |
| 📡 **Data Pipeline** | Custom ETL (`live_data_pipeline.py`, `process_real_data.py`) |
| ☁️ **Deployment** | Streamlit Community Cloud |
| 🔄 **CI/CD** | GitHub auto-deploy on push |

</div>

---

## 📁 Project Structure

```
Pharma-Drug-Demand-Forecasting/
│
├── 📄 app.py                          # 🏠 Main entry point (home dashboard)
├── 📄 forecasting.py                  # 📊 Core forecasting engine
├── 📄 bayesian_forecasting.py         # 🔬 Hierarchical Bayesian model
├── 📄 state_forecasting.py            # 🗺️ State-level forecasts
├── 📄 sensitivity_analysis.py         # 🎯 Sensitivity & scenario logic
├── 📄 process_real_data.py            # 🧹 Real-world data ingestion
├── 📄 process_state_data.py           # 🏥 State + hospital processing
├── 📄 live_data_pipeline.py           # 📡 Live data fetching pipeline
├── 📄 run_all.py                      # 🚀 End-to-end orchestrator
│
├── 📁 pages/                          # Multi-page Streamlit app
│   ├── 1_📈_National_Forecast.py
│   ├── 2_📖_State_Explorer.py
│   ├── 3_🔬_Bayesian.py
│   ├── 4_🎯_Sensitivity.py
│   └── 5_📋_Executive_Summary.py
│
├── 📁 data/                           # Real-world datasets
│   ├── hospital_directory.csv         # Hospital infrastructure data
│   ├── hospitals_by_state.csv
│   ├── ihme_diabetes_processed.csv    # IHME GBD 2023 processed
│   ├── IHME-GBD_2023_DATA.csv         # Raw IHME Global Burden of Disease
│   ├── india_diabetes_real_master.csv
│   ├── India_dataset_population.csv
│   ├── live_wb_health_exp.csv         # World Bank health expenditure
│   ├── live_wb_population.csv         # World Bank population
│   ├── live_who_diabetes.csv          # WHO diabetes indicators
│   ├── NFHS_5_Factsheets_Data.xls     # NFHS-5 health survey
│   ├── state_master.csv
│   └── state_prevalence_icmr.csv      # ICMR state-level prevalence
│
├── 📁 outputs/                        # Generated artifacts
│   ├── bayesian_posteriors.png        # Posterior distribution plots
│   ├── bayesian_state_intervals.csv   # State-level credible intervals
│   ├── demand_plot.png
│   ├── demand_with_confidence.png
│   ├── forecast_combined_state_year.csv
│   ├── forecast_output.csv
│   ├── forecast_output_advanced.csv
│   ├── patient_funnel.png
│   ├── revenue_plot.png
│   ├── scenario_comparison.png
│   ├── sensitivity_analysis.csv
│   └── tornado_chart.png
│
├── 📁 .streamlit/
│   └── config.toml                    # Custom theme & runtime config
│
├── 📋 requirements.txt                # Pinned dependencies
├── 📋 runtime.txt                     # Python version lock
├── 📋 .gitignore
└── 📖 README.md                       # You are here!
```

---

## 🏗️ Methodology

### 🧮 Hierarchical Bayesian Model

```
┌────────────────────────────────────────────────────────┐
│ LEVEL 1 (National)                                     │
│ μ ~ Normal(prior_mean, prior_sd)                       │
│ Prior on overall mean diabetes prevalence              │
└────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ LEVEL 2      │  │ LEVEL 2      │  │ LEVEL 2      │
│ Region       │  │ Region       │  │ Region       │
│ (N/S/E/W/    │  │ offset_r     │  │ Central/NE)  │
│  Central/NE) │  │              │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ LEVEL 3      │  │ LEVEL 3      │  │ LEVEL 3      │
│ State        │  │ State        │  │ State        │
│ deviation_s  │  │ deviation_s  │  │ deviation_s  │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Why Hierarchical?
States within the same region **share statistical strength** while preserving local variation. Data-sparse states (e.g., Northeast) borrow information from data-rich neighbors **without overfitting**.

### Why Bayesian?
Point estimates lie. Real pharma decisions need **uncertainty bounds**. Every state-level forecast comes with a **95% credible interval** — actionable for procurement, distribution, and investment.

### Sampler Engineering
- **nutpie (Rust NUTS)**: 5–10× faster than pure-Python PyMC
- **Non-centered parameterization**: Eliminates funnel pathologies
- **Logit-scale modeling**: Numerical stability for prevalence ∈ (0,1)
- **4 chains × 1000 draws + 500 tuning**: Convergence verified via R̂ < 1.01

---

## 📡 Data Sources

| Source | Description | File |
|:---|:---|:---|
| 🏥 **ICMR** | India's state-level diabetes prevalence | `state_prevalence_icmr.csv` |
| 🌍 **IHME GBD 2023** | Global Burden of Disease — diabetes metrics | `IHME-GBD_2023_DATA.csv` |
| 🏥 **NFHS-5** | National Family Health Survey factsheets | `NFHS_5_Factsheets_Data.xls` |
| 💊 **WHO GHO** | Diabetes indicators | `live_who_diabetes.csv` |
| 💰 **World Bank** | Health expenditure & population | `live_wb_health_exp.csv`, `live_wb_population.csv` |
| 🏨 **Hospital Registry** | Infrastructure mapping | `hospital_directory.csv` |

---

## 🚀 Quick Start

### 🌐 Use the Live App (Zero Setup)
👉 **https://pharma-forecasting.streamlit.app/**

### 💻 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/mohammadarkam077-beep/Pharma-Drug-Demand-Forecasting.git
cd Pharma-Drug-Demand-Forecasting

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Run full pipeline to regenerate outputs
python run_all.py

# 5. Launch the app
streamlit run app.py
```

App opens at **`http://localhost:8501`** 🎉

---

## 🔄 Reproducing the Full Pipeline

```bash
# Stage 1: Ingest & clean real-world data
python process_real_data.py
python process_state_data.py

# Stage 2: Run forecasts
python forecasting.py
python state_forecasting.py

# Stage 3: Bayesian inference (slowest stage, ~5–10 min)
python bayesian_forecasting.py

# Stage 4: Sensitivity analysis
python sensitivity_analysis.py

# OR — run everything end-to-end:
python run_all.py
```

All artifacts (CSVs, PNGs) land in `outputs/`.

---

## 📦 Dependencies

```txt
streamlit==1.39.0
pandas==2.2.3
numpy==1.26.4
plotly==5.24.1
pillow==10.4.0
openpyxl==3.1.5
```

Python: **3.11** (locked via `runtime.txt`)

---

## 🎓 Engineering Highlights

- ✅ **Real data pipeline** — not synthetic; pulls from ICMR/WHO/IHME/NFHS/World Bank
- ✅ **Hierarchical Bayesian inference** with proper uncertainty propagation
- ✅ **Production deployment** — CI/CD via GitHub → Streamlit Cloud
- ✅ **Multi-page architecture** — clean separation of concerns
- ✅ **Reproducible pipeline** — single-command end-to-end execution
- ✅ **Pinned dependencies** + Python version lock for deterministic builds
- ✅ **Custom theming** via `.streamlit/config.toml`
- ✅ **Modular Python** — separate modules for forecasting, Bayesian, sensitivity, ETL

---

## 📈 Sample Insights from the Platform

> 🔥 **Untapped opportunity:** ₹24,466 Cr (280% larger than current market)  
> 📍 **Concentration risk:** Top 5 states = 46% of total revenue  
> 🎯 **Diagnosis gap:** Only 31% of India's diabetics are currently diagnosed  
> 🚀 **Growth trajectory:** +96% projected over the next 5 years  
> 🏥 **Infrastructure correlation:** Hospital density strongly predicts diagnosed-prevalence ratio

---

## 🔮 Roadmap

- [ ] Extend beyond diabetes → cardiovascular, oncology, respiratory
- [ ] Real-time prescription data integration (PMJAY, e-Aushadhi)
- [ ] Scenario simulator (what-if pricing/policy changes)
- [ ] PDF export for executive reports
- [ ] Multi-language support (Hindi, Tamil, Bengali, Marathi)
- [ ] REST API endpoints for programmatic forecast access
- [ ] District-level granularity (currently state-level)
- [ ] Time-varying covariates (urbanization, BMI trends)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

```bash
1. Fork the repo
2. git checkout -b feature/your-feature
3. git commit -m "Add: your feature"
4. git push origin feature/your-feature
5. Open a Pull Request
```

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Mohammad Arkam**  
*M.Pharm* | Pharma Analytics | Generative AI*

[![GitHub](https://img.shields.io/badge/GitHub-mohammadarkam077--beep-181717?style=for-the-badge&logo=github)](https://github.com/mohammadarkam077-beep)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/YOUR-LINKEDIN-HANDLE)

---

## 🙏 Acknowledgments

- **Indian Council of Medical Research (ICMR)** — state-level diabetes prevalence data
- **Institute for Health Metrics and Evaluation (IHME)** — GBD 2023 estimates
- **World Health Organization (WHO)** — Global Health Observatory
- **Ministry of Health & Family Welfare (NFHS-5)** — National Family Health Survey
- **World Bank** — Health expenditure & demographic indicators
- **Streamlit** — for making data apps accessible
- **PyMC + nutpie** teams — for state-of-the-art Bayesian tooling

---

<div align="center">

### ⭐ If this project helped you, please give it a star!

### 🔗 [**Live Demo**](https://pharma-forecasting.streamlit.app/) · [**Report Bug**](https://github.com/mohammadarkam077-beep/Pharma-Drug-Demand-Forecasting/issues) · [**Request Feature**](https://github.com/mohammadarkam077-beep/Pharma-Drug-Demand-Forecasting/issues)

<br>

**Made with ❤️ in India 🇮🇳 by Mohammad Arkam**

</div>
