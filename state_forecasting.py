"""
State-wise Forecasting Engine
==============================
Combines national time series forecast with state-level distributions
to create state × year forecasts.
"""

import pandas as pd
import numpy as np

def main():
    print("="*60)
    print("🌐 STATE × YEAR FORECASTING ENGINE")
    print("="*60)
    
    # Load national time series forecast
    national = pd.read_csv(r"D:\Forecasting\outputs\forecast_output_advanced.csv")
    print(f"✓ Loaded national forecast: {len(national)} years")
    
    # Load state master
    states = pd.read_csv(r"D:\Forecasting\data\state_master.csv")
    print(f"✓ Loaded state master: {len(states)} states")
    
    # Calculate each state's share of national diabetics
    states['Share_of_India'] = states['Total_Diabetics'] / states['Total_Diabetics'].sum()
    
    print(f"\n🔄 Generating state × year forecasts...")
    
    # Cross join: every state × every year
    combined = []
    
    for _, state_row in states.iterrows():
        for _, year_row in national.iterrows():
            combined.append({
                # Time dimension
                'Year': year_row['Year'],
                'Is_Forecast': year_row.get('Is_Forecast', 0),
                
                # State dimension
                'State': state_row['State'],
                'Region': state_row['Region'],
                'Tier': state_row.get('Tier', 'Mixed'),
                'Market_Tier': state_row['Market_Tier'],
                
                # Population (state share)
                'State_Population': state_row['Population_2024'] * (year_row['Population'] / national['Population'].iloc[-1]),
                
                # Demand (state share)
                'State_Demand': year_row['Annual_Demand'] * state_row['Share_of_India'],
                'State_Demand_M': year_row['Annual_Demand'] * state_row['Share_of_India'] / 1e6,
                
                # Revenue (state share)
                'State_Revenue': year_row['Revenue'] * state_row['Share_of_India'],
                'State_Revenue_Cr': year_row['Revenue'] * state_row['Share_of_India'] / 1e7,
                
                # Patients (state share)
                'State_Patients': year_row['Patients'] * state_row['Share_of_India'],
                
                # State-specific metrics
                'State_Prevalence': state_row['Diabetes_Prevalence_2021'],
                'State_Diagnosis_Rate': state_row['Diagnosis_Rate_Adjusted'],
                'State_Treatment_Rate': state_row['Treatment_Rate_Adjusted'],
                'State_Healthcare_Score': state_row['Healthcare_Access_Score'],
                'State_Hospitals': state_row['Total_Hospitals'],
                'State_Priority_Score': state_row['Priority_Score'],
                
                # Opportunity
                'Treatment_Gap_M': state_row['Treatment_Gap'] / 1e6,
                'Opportunity_Cr': state_row['Opportunity_Revenue_Cr']
            })
    
    df = pd.DataFrame(combined)
    
    # Sort
    df = df.sort_values(['Year', 'State_Revenue_Cr'], ascending=[True, False]).reset_index(drop=True)
    
    # Save
    output = r"D:\Forecasting\outputs\forecast_combined_state_year.csv"
    df.to_csv(output, index=False)
    
    print(f"\n✅ Saved: {output}")
    print(f"   {len(df):,} rows ({len(states)} states × {len(national)} years)")
    print(f"   {len(df.columns)} columns")
    
    # Summary
    print("\n📊 SAMPLE: 2024 Top 5 States")
    print("="*60)
    sample = df[df['Year'] == 2024].head()[['State', 'Region', 'State_Patients', 'State_Revenue_Cr', 'State_Priority_Score']]
    print(sample.to_string(index=False))
    
    return df


if __name__ == "__main__":
    df = main()