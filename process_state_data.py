"""
Master State-Level Data Integrator
====================================
Combines:
1. ICMR-INDIAB-17 state prevalence
2. Hospital directory (30,273 hospitals)
3. State population estimates
4. Healthcare access scores

Output: state_master.csv — ready for state-wise forecasting
"""

import pandas as pd
import numpy as np

# ============================================================
# STATE POPULATION DATA (Census 2011 + Projections)
# ============================================================
STATE_POPULATION_2024 = {
    'Uttar Pradesh': 240e6, 'Maharashtra': 126e6, 'Bihar': 130e6,
    'West Bengal': 99e6, 'Madhya Pradesh': 87e6, 'Tamil Nadu': 78e6,
    'Rajasthan': 82e6, 'Karnataka': 68e6, 'Gujarat': 72e6,
    'Andhra Pradesh': 53e6, 'Odisha': 47e6, 'Telangana': 39e6,
    'Kerala': 36e6, 'Jharkhand': 39e6, 'Assam': 36e6,
    'Punjab': 31e6, 'Chhattisgarh': 30e6, 'Haryana': 30e6,
    'Delhi': 21e6, 'Jammu and Kashmir': 14e6, 'Uttarakhand': 12e6,
    'Himachal Pradesh': 7.5e6, 'Tripura': 4.2e6, 'Meghalaya': 3.5e6,
    'Manipur': 3.2e6, 'Nagaland': 2.3e6, 'Goa': 1.6e6,
    'Arunachal Pradesh': 1.6e6, 'Puducherry': 1.6e6, 'Mizoram': 1.3e6,
    'Chandigarh': 1.2e6, 'Sikkim': 0.7e6
}


def load_icmr_state_data(path):
    """Load ICMR-INDIAB-17 state prevalence data"""
    print(f"📥 Loading ICMR state data...")
    df = pd.read_csv(path)
    print(f"  ✓ {len(df)} states with prevalence data")
    print(f"  ✓ Range: {df['Diabetes_Prevalence_2021'].min():.1%} - {df['Diabetes_Prevalence_2021'].max():.1%}")
    return df


def load_hospital_data(path):
    """Load hospital directory and aggregate by state"""
    print(f"\n📥 Loading hospital directory...")
    
    try:
        df = pd.read_csv(path, encoding='latin1', low_memory=False)
    except:
        df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    
    print(f"  ✓ {len(df):,} hospitals loaded")
    
    # Find the State column (it might have different naming)
    state_col = None
    for col in ['State', 'state', 'STATE', 'State_Name']:
        if col in df.columns:
            state_col = col
            break
    
    if state_col is None:
        print(f"  ⚠ No State column found. Available columns:")
        print(f"  {list(df.columns)}")
        return None
    
    # Standardize state names
    df[state_col] = df[state_col].str.strip().str.title()
    
    # Aggregate
    hospital_counts = df.groupby(state_col).agg(
        Total_Hospitals=('Hospital_Name', 'count') if 'Hospital_Name' in df.columns else (state_col, 'count')
    ).reset_index()
    
    hospital_counts.columns = ['State', 'Total_Hospitals']
    
    # Standardize state names to match ICMR
    state_mapping = {
        'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
        'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
        'Jammu & Kashmir': 'Jammu and Kashmir',
    }
    hospital_counts['State'] = hospital_counts['State'].replace(state_mapping)
    
    print(f"  ✓ Aggregated to {len(hospital_counts)} states")
    print(f"  Top 5 states:")
    print(hospital_counts.nlargest(5, 'Total_Hospitals').to_string(index=False))
    
    return hospital_counts


def calculate_healthcare_access(state_df):
    """
    Calculate healthcare access score:
    - Hospitals per million population
    - Used as a proxy for diagnosis & treatment rates
    """
    print(f"\n🏥 Calculating healthcare access scores...")
    
    # Hospitals per million
    state_df['Hospitals_per_Million'] = state_df['Total_Hospitals'] / (state_df['Population_2024'] / 1e6)
    
    # Normalize 0-1 (min-max scaling)
    min_h = state_df['Hospitals_per_Million'].min()
    max_h = state_df['Hospitals_per_Million'].max()
    state_df['Healthcare_Access_Score'] = (state_df['Hospitals_per_Million'] - min_h) / (max_h - min_h)
    
    # Adjust diagnosis rate based on healthcare access
    # National average: 56% (NFHS-5)
    # High access states get +10%, low access -10%
    base_diagnosis = 0.56
    state_df['Diagnosis_Rate_Adjusted'] = base_diagnosis + (state_df['Healthcare_Access_Score'] - 0.5) * 0.30
    state_df['Diagnosis_Rate_Adjusted'] = state_df['Diagnosis_Rate_Adjusted'].clip(0.30, 0.85)
    
    # Adjust treatment rate similarly
    base_treatment = 0.64
    state_df['Treatment_Rate_Adjusted'] = base_treatment + (state_df['Healthcare_Access_Score'] - 0.5) * 0.25
    state_df['Treatment_Rate_Adjusted'] = state_df['Treatment_Rate_Adjusted'].clip(0.40, 0.85)
    
    print(f"  ✓ Diagnosis rate range: {state_df['Diagnosis_Rate_Adjusted'].min():.2f} - {state_df['Diagnosis_Rate_Adjusted'].max():.2f}")
    print(f"  ✓ Treatment rate range: {state_df['Treatment_Rate_Adjusted'].min():.2f} - {state_df['Treatment_Rate_Adjusted'].max():.2f}")
    
    return state_df


def calculate_diabetes_burden(state_df):
    """Calculate state-wise diabetes patient counts"""
    print(f"\n📊 Calculating disease burden by state...")
    
    # Adult population (62% of total)
    state_df['Adult_Pop'] = state_df['Population_2024'] * 0.62
    
    # Total diabetic population (using ICMR prevalence)
    state_df['Total_Diabetics'] = state_df['Adult_Pop'] * state_df['Diabetes_Prevalence_2021']
    
    # Diagnosed patients
    state_df['Diagnosed_Patients'] = state_df['Total_Diabetics'] * state_df['Diagnosis_Rate_Adjusted']
    
    # On treatment
    state_df['Treated_Patients'] = state_df['Diagnosed_Patients'] * state_df['Treatment_Rate_Adjusted']
    
    # Compliance (assume 78% national average, can be refined)
    state_df['Compliant_Patients'] = state_df['Treated_Patients'] * 0.78
    
    # Treatment gap (untapped market!)
    state_df['Treatment_Gap'] = state_df['Total_Diabetics'] - state_df['Treated_Patients']
    state_df['Treatment_Gap_Pct'] = state_df['Treatment_Gap'] / state_df['Total_Diabetics']
    
    print(f"  ✓ Total India diabetics: {state_df['Total_Diabetics'].sum()/1e6:.1f}M")
    print(f"  ✓ On treatment: {state_df['Treated_Patients'].sum()/1e6:.1f}M")
    print(f"  ✓ Treatment gap: {state_df['Treatment_Gap'].sum()/1e6:.1f}M (untapped market!)")
    
    return state_df


def calculate_market_opportunity(state_df, price_per_unit=12, units_per_year=365):
    """Calculate revenue opportunity by state"""
    print(f"\n💰 Calculating market opportunity by state...")
    
    # Annual demand (units)
    state_df['Annual_Demand'] = state_df['Compliant_Patients'] * units_per_year
    
    # Revenue (₹)
    state_df['Annual_Revenue'] = state_df['Annual_Demand'] * price_per_unit
    state_df['Revenue_Cr'] = state_df['Annual_Revenue'] / 1e7
    
    # Opportunity (if treatment gap was closed)
    state_df['Opportunity_Patients'] = state_df['Treatment_Gap']
    state_df['Opportunity_Revenue_Cr'] = (state_df['Opportunity_Patients'] * units_per_year * price_per_unit) / 1e7
    
    # Tier classification
    state_df['Market_Tier'] = pd.cut(
        state_df['Revenue_Cr'],
        bins=[0, 100, 500, 1500, 5000, np.inf],
        labels=['Tier-5 (Small)', 'Tier-4 (Emerging)', 'Tier-3 (Mid)', 'Tier-2 (Major)', 'Tier-1 (Mega)']
    )
    
    # Priority score (higher = better target)
    # Combines current revenue + opportunity + healthcare access
    state_df['Priority_Score'] = (
        (state_df['Revenue_Cr'] / state_df['Revenue_Cr'].max()) * 0.4 +
        (state_df['Opportunity_Revenue_Cr'] / state_df['Opportunity_Revenue_Cr'].max()) * 0.4 +
        state_df['Healthcare_Access_Score'] * 0.2
    ) * 100
    
    print(f"  ✓ Total revenue (current): ₹{state_df['Revenue_Cr'].sum():,.0f} Cr")
    print(f"  ✓ Untapped opportunity: ₹{state_df['Opportunity_Revenue_Cr'].sum():,.0f} Cr")
    
    return state_df


def main():
    print("="*60)
    print("🌐 MASTER STATE DATA INTEGRATOR")
    print("="*60)
    
    # Load all data sources
    icmr = load_icmr_state_data(r"D:\Forecasting\data\state_prevalence_icmr.csv")
    hospitals = load_hospital_data(r"D:\Forecasting\data\hospital_directory.csv")
    
    # Add population
    icmr['Population_2024'] = icmr['State'].map(STATE_POPULATION_2024)
    
    # Merge with hospital data
    if hospitals is not None:
        master = icmr.merge(hospitals, on='State', how='left')
        # Fill NaN hospitals with state average
        master['Total_Hospitals'] = master['Total_Hospitals'].fillna(
            master['Total_Hospitals'].median()
        )
    else:
        # Fallback if hospital data fails
        print("⚠ Using estimated hospital counts...")
        master = icmr.copy()
        master['Total_Hospitals'] = master['Population_2024'] / 50000  # ~20 hospitals per million
    
    # Drop states without population data
    master = master.dropna(subset=['Population_2024'])
    
    # Calculate all metrics
    master = calculate_healthcare_access(master)
    master = calculate_diabetes_burden(master)
    master = calculate_market_opportunity(master)
    
    # Sort by priority
    master = master.sort_values('Priority_Score', ascending=False).reset_index(drop=True)
    master['Rank'] = range(1, len(master) + 1)
    
    # Save
    output_path = r"D:\Forecasting\data\state_master.csv"
    master.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("🏆 TOP 10 STATES BY PRIORITY SCORE")
    print("="*60)
    cols_to_show = ['Rank', 'State', 'Region', 'Diabetes_Prevalence_2021', 
                    'Total_Diabetics', 'Revenue_Cr', 'Opportunity_Revenue_Cr', 
                    'Market_Tier', 'Priority_Score']
    print(master[cols_to_show].head(10).to_string(index=False))
    
    print(f"\n✅ Saved: {output_path}")
    print(f"   {len(master)} states × {len(master.columns)} columns")
    
    return master


if __name__ == "__main__":
    df = main()