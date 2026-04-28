import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# Load Data
# ----------------------------
data_path = r"D:\Forecasting\data\India_dataset_population.csv"
df = pd.read_csv(data_path, encoding="latin1")

# ----------------------------
# Parameters
# ----------------------------
df["prevalence"] = (0.06 + (df["Year"] - 2010) * 0.001).clip(upper=0.12)
df["diagnosis_rate"] = 0.7
df["treatment_rate"] = 0.6
df["market_share"] = (0.1 + (df["Year"] - 2010) * 0.01).clip(upper=0.5)
df["compliance"] = 0.8

# Population segmentation
df["Elderly_Pop"] = df["Population"] * 0.2
df["Adult_Pop"] = df["Population"] * 0.5

# ----------------------------
# Demand Calculation
# ----------------------------
df["Patients"] = (
    df["Population"]
    * df["prevalence"]
    * df["diagnosis_rate"]
    * df["treatment_rate"]
    * df["market_share"]
    * df["compliance"]
)

df["Annual_Demand"] = df["Patients"] * 365

# ----------------------------
# Scenario Analysis
# ----------------------------
scenarios = {"Worst": 0.15, "Base": 0.25, "Best": 0.35}

for name, share in scenarios.items():
    df[f"Demand_{name}"] = (
        df["Population"]
        * df["prevalence"]
        * df["diagnosis_rate"]
        * df["treatment_rate"]
        * share
        * df["compliance"]
        * 365
    )

# ----------------------------
# Revenue
# ----------------------------
price_per_unit = 12
df["Revenue"] = df["Annual_Demand"] * price_per_unit

# ----------------------------
# Monte Carlo Simulation
# ----------------------------
simulations = []

for _ in range(100):
    prevalence_sim = np.random.normal(0.08, 0.01)
    market_sim = np.random.normal(0.25, 0.05)

    demand = (
        df["Population"]
        * prevalence_sim
        * df["diagnosis_rate"]
        * df["treatment_rate"]
        * market_sim
        * df["compliance"]
        * 365
    )

    simulations.append(demand)

df["Demand_Mean"] = np.mean(simulations, axis=0)

# ----------------------------
# Metrics
# ----------------------------
start = df["Annual_Demand"].iloc[0]
end = df["Annual_Demand"].iloc[-1]
years = len(df) - 1

cagr = (end / start) ** (1/years) - 1

# ----------------------------
# Output Formatting
# ----------------------------
df["Annual_Demand_M"] = (df["Annual_Demand"] / 1e6).round(2)
df["Revenue_Cr"] = (df["Revenue"] / 1e7).round(2)

print("\n=== SUMMARY ===")
print(f"Start Demand: {start/1e9:.2f} B units")
print(f"End Demand: {end/1e9:.2f} B units")
print(f"Growth: {(end-start)/1e9:.2f} B units")
print(f"CAGR: {cagr:.2%}")

print("High growth market opportunity" if cagr > 0.05 else "Moderate growth market")
print("Recommendation: Increase production capacity and focus on high-growth segments.")

# ----------------------------
# Save Outputs
# ----------------------------
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/forecast_output.csv", index=False)

# Demand plot
plt.figure()
plt.plot(df["Year"], df["Annual_Demand"])
plt.title("Drug Demand Forecast")
plt.xlabel("Year")
plt.ylabel("Units")
plt.savefig("outputs/demand_plot.png")

# Revenue plot
plt.figure()
plt.plot(df["Year"], df["Revenue"] / 1e9)
plt.title("Revenue Forecast (₹ Billion)")
plt.xlabel("Year")
plt.ylabel("₹ Billion")
plt.savefig("outputs/revenue_plot.png")

# ----------------------------
# Final Checks
# ----------------------------
assert (df["Annual_Demand"] >= 0).all(), "Negative demand detected"

peak_year = df.loc[df["Annual_Demand"].idxmax(), "Year"]
print(f"Peak demand year: {peak_year}")
print("Units: Demand in Billions, Revenue in Crores (INR)")