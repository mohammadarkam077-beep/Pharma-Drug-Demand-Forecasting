"""
Sensitivity Analysis with Tornado Charts (FIXED)
==================================================
Shows which parameters most impact revenue forecasts using
realistic parameter-specific uncertainty ranges.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SensitivityAnalyzer:
    """
    Performs sensitivity testing using realistic per-parameter uncertainty
    ranges (based on data source confidence) to identify highest-impact drivers.
    """

    BASE_PARAMS = {
        'prevalence':     0.114,
        'diagnosis_rate': 0.62,
        'treatment_rate': 0.68,
        'compliance':     0.78,
        'market_share':   0.40,
        'price_per_unit': 12,
        'population':     1.438e9,
        'adult_share':    0.62,
    }

    # Realistic uncertainty ranges per parameter (based on data source quality)
    # Format: (low_multiplier, high_multiplier)
    PARAM_RANGES = {
        'prevalence':     (0.90, 1.15),   # ICMR-INDIAB: ±10–15% (well-studied)
        'diagnosis_rate': (0.75, 1.20),   # NFHS-5: ±20–25% (regional variance)
        'treatment_rate': (0.80, 1.15),   # Healthcare access varies
        'compliance':     (0.70, 1.10),   # Highly variable, hard to measure
        'market_share':   (0.60, 1.40),   # Most volatile (competition, marketing)
        'price_per_unit': (0.85, 1.15),   # NPPA-regulated in India
        'population':     (0.98, 1.02),   # UN/Census: very accurate
        'adult_share':    (0.97, 1.03),   # Demographic data: very stable
    }

    # Human-readable labels for charts
    PARAM_LABELS = {
        'prevalence':     'Diabetes Prevalence',
        'diagnosis_rate': 'Diagnosis Rate',
        'treatment_rate': 'Treatment Rate',
        'compliance':     'Patient Compliance',
        'market_share':   'Market Share',
        'price_per_unit': 'Price per Unit',
        'population':     'Population',
        'adult_share':    'Adult Population Share',
    }

    def __init__(self, base_params=None, param_ranges=None):
        self.params = base_params or self.BASE_PARAMS.copy()
        self.ranges = param_ranges or self.PARAM_RANGES.copy()
        self.results = None

    def calculate_revenue(self, params):
        """Standard revenue formula (₹)"""
        adult_pop = params['population'] * params['adult_share']
        patients = (adult_pop
                    * params['prevalence']
                    * params['diagnosis_rate']
                    * params['treatment_rate']
                    * params['market_share']
                    * params['compliance'])
        annual_demand = patients * 365
        revenue = annual_demand * params['price_per_unit']
        return revenue

    def run_analysis(self):
        """One-at-a-time (OAT) sensitivity using parameter-specific ranges"""
        base_revenue = self.calculate_revenue(self.params)
        base_revenue_cr = base_revenue / 1e7
        print(f"📊 Base Revenue: ₹{base_revenue_cr:,.2f} Cr\n")

        rows = []
        for param_name, base_value in self.params.items():
            low_mult, high_mult = self.ranges[param_name]

            # LOW scenario
            low_params = self.params.copy()
            low_params[param_name] = base_value * low_mult
            low_revenue = self.calculate_revenue(low_params)

            # HIGH scenario
            high_params = self.params.copy()
            high_params[param_name] = base_value * high_mult
            high_revenue = self.calculate_revenue(high_params)

            # Impact in Crores (deviation from base)
            low_impact_cr = (low_revenue - base_revenue) / 1e7
            high_impact_cr = (high_revenue - base_revenue) / 1e7

            # Total swing = full range of revenue change
            total_range_cr = abs(high_impact_cr - low_impact_cr)

            # Elasticity-like sensitivity: % revenue change per % param change
            param_pct_change = (high_mult - low_mult) * 100
            revenue_pct_change = total_range_cr / base_revenue_cr * 100
            elasticity = revenue_pct_change / param_pct_change if param_pct_change else 0

            rows.append({
                'Parameter':         param_name,
                'Label':             self.PARAM_LABELS.get(param_name, param_name),
                'Base_Value':        base_value,
                'Low_Multiplier':    low_mult,
                'High_Multiplier':   high_mult,
                'Param_Range_%':     param_pct_change,
                'Low_Revenue_Cr':    low_revenue / 1e7,
                'High_Revenue_Cr':   high_revenue / 1e7,
                'Low_Impact_Cr':     low_impact_cr,
                'High_Impact_Cr':    high_impact_cr,
                'Total_Range_Cr':    total_range_cr,
                'Revenue_Range_%':   revenue_pct_change,
                'Elasticity':        elasticity,
            })

        self.results = pd.DataFrame(rows).sort_values(
            'Total_Range_Cr', ascending=False
        ).reset_index(drop=True)

        print("🎯 SENSITIVITY RANKING (realistic per-parameter ranges):\n")
        display_cols = ['Label', 'Param_Range_%', 'Total_Range_Cr',
                        'Revenue_Range_%', 'Elasticity']
        print(self.results[display_cols].to_string(index=False,
              formatters={
                  'Param_Range_%':   '{:6.1f}%'.format,
                  'Total_Range_Cr':  '₹{:,.1f}'.format,
                  'Revenue_Range_%': '{:6.1f}%'.format,
                  'Elasticity':      '{:.3f}'.format,
              }))
        return self

    def plot_tornado(self, output_path='outputs/tornado_chart.png'):
        """Create tornado chart sorted by impact magnitude"""
        df = self.results.sort_values('Total_Range_Cr').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(13, 7))
        y_pos = np.arange(len(df))

        # Bars
        ax.barh(y_pos, df['Low_Impact_Cr'],  color='#EF4444', alpha=0.85,
                label='Downside scenario', edgecolor='darkred', linewidth=0.8)
        ax.barh(y_pos, df['High_Impact_Cr'], color='#10B981', alpha=0.85,
                label='Upside scenario',   edgecolor='darkgreen', linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Label'], fontsize=11)
        ax.set_xlabel('Revenue Impact (₹ Crores)', fontsize=11)
        ax.set_title(
            'Sensitivity Analysis — Tornado Chart\n'
            'Revenue impact based on realistic per-parameter uncertainty',
            fontsize=13, fontweight='bold', pad=15
        )
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Dynamic offset for labels (5% of max impact)
        max_abs = max(df['High_Impact_Cr'].abs().max(),
                      df['Low_Impact_Cr'].abs().max())
        offset = max_abs * 0.02

        for i, row in df.iterrows():
            # Low side label
            ax.text(row['Low_Impact_Cr'] - offset, i,
                    f"₹{row['Low_Impact_Cr']:,.0f} Cr\n({row['Low_Multiplier']:.0%})",
                    ha='right', va='center', fontsize=8, color='darkred')
            # High side label
            ax.text(row['High_Impact_Cr'] + offset, i,
                    f"+₹{row['High_Impact_Cr']:,.0f} Cr\n({row['High_Multiplier']:.0%})",
                    ha='left', va='center', fontsize=8, color='darkgreen')

        # Padding so labels don't get clipped
        ax.set_xlim(df['Low_Impact_Cr'].min() * 1.35,
                    df['High_Impact_Cr'].max() * 1.35)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved tornado chart: {output_path}")
        return self


# === USAGE ===
if __name__ == "__main__":
    import os
    os.makedirs(r"D:\Forecasting\outputs", exist_ok=True)

    analyzer = (SensitivityAnalyzer()
                .run_analysis()
                .plot_tornado(r"D:\Forecasting\outputs\tornado_chart.png"))

    analyzer.results.to_csv(
        r"D:\Forecasting\outputs\sensitivity_analysis.csv",
        index=False
    )
    print(f"✓ Saved data: D:\\Forecasting\\outputs\\sensitivity_analysis.csv")

    # Top driver insight
    top = analyzer.results.iloc[0]
    print(f"\n💡 KEY INSIGHT:")
    print(f"   '{top['Label']}' is the highest-impact driver:")
    print(f"   ±{top['Param_Range_%']/2:.1f}% variation → ₹{top['Total_Range_Cr']:,.0f} Cr swing")
    print(f"   (Elasticity: {top['Elasticity']:.2f} — "
          f"{'highly sensitive' if top['Elasticity'] > 1 else 'moderately sensitive'})")