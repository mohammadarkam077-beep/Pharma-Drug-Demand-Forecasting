import pandas as pd

# Load file
df = pd.read_csv(r"D:\Forecasting\outputs\forecast_output.csv")

# Reset index
df.reset_index(drop=True, inplace=True)

# Clean decimals
df = df.round(4)

# Add business columns
df["Annual_Demand_M"] = (df["Annual_Demand"] / 1e6).round(2)
df["Revenue_Cr"] = (df["Revenue"] / 1e7).round(2)

# 🔥 THIS LINE CREATES THE FILE
df.to_csv(r"D:\Forecasting\outputs\forecast_output_clean.csv", index=False)

print("✅ Clean file saved successfully")