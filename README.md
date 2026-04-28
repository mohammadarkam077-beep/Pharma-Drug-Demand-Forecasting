# 📊 Pharmaceutical Drug Demand Forecasting (Python)

## 📌 Overview

This project builds a **patient-based pharmaceutical demand forecasting model** using population data and epidemiological assumptions.
It estimates **drug demand, revenue, growth trends, and business insights** for a chronic disease (e.g., diabetes) in India.

---

## 🎯 Objective

To forecast:

* Annual drug demand 📦
* Revenue potential 💰
* Market growth 📈

using a **data-driven and scenario-based approach**

---

## 🧠 Methodology

### 1. Population Data

* Year-wise population dataset (India)

### 2. Epidemiological Parameters

* Prevalence (time-varying)
* Diagnosis rate
* Treatment rate
* Market share (growth-based)
* Patient compliance

### 3. Demand Calculation

Patients → Daily dosage → Annual demand

### 4. Revenue Estimation

Revenue = Demand × Price per unit

### 5. Advanced Features

* Scenario analysis (Worst / Base / Best)
* Monte Carlo simulation (uncertainty modeling)
* Sensitivity analysis
* CAGR (growth rate calculation)
* Peak demand identification

---

## 📊 Key Results

* **Demand Growth:** 0.91B → 3.13B units
* **Revenue Growth:** ₹10.8B → ₹37.5B
* **CAGR:** ~9.27%
* **Peak Demand Year:** 2024

---

## 💡 Insights

* Growth driven by **increasing patient pool and treatment penetration**
* Market shows **high-growth potential**
* Strong opportunity for **capacity expansion**

---

## 📌 Business Recommendation

> Increase production capacity and focus on high-growth therapeutic segments to capture expanding demand.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib

---

## 📁 Project Structure

```
Pharma-Drug-Demand-Forecasting/
│
├── forecasting.py
├── data/
│   └── India_dataset_population.csv
├── outputs/
│   ├── forecast_output.csv
│   ├── demand_plot.png
│   └── revenue_plot.png
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/Pharma-Drug-Demand-Forecasting.git
cd Pharma-Drug-Demand-Forecasting
pip install -r requirements.txt
python forecasting.py
```

---

## 📈 Output

* 📊 Demand forecast plot
* 💰 Revenue forecast plot
* 📁 CSV file with full forecast data

---

## 🔮 Future Improvements

* Integrate real-world datasets (WHO, IHME)
* Add regional segmentation
* Build dashboard (Power BI / Tableau)
* Include pricing dynamics and competition

---

## 👤 Author

**Md Arkam**
Pharmaceutics + Data Analytics

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!
