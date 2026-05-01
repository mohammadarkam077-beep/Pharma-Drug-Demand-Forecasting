"""
Live Data Pipeline
==================
Auto-fetches latest data from WHO, World Bank, data.gov.in
"""

import requests
import pandas as pd
from datetime import datetime
import time


class LiveDataPipeline:
    """Auto-refreshes pharma forecasting data from APIs"""
    
    def fetch_who_diabetes(self, country='IND'):
        """WHO Global Health Observatory"""
        url = f"https://ghoapi.azureedge.net/api/NCD_GLUC_04?$filter=SpatialDim eq '{country}'"
        try:
            data = requests.get(url, timeout=10).json()
            df = pd.DataFrame(data['value'])
            df['fetched_at'] = datetime.now()
            print(f"✓ WHO: {len(df)} records")
            return df
        except Exception as e:
            print(f"⚠ WHO failed: {e}")
            return None
    
    def fetch_world_bank_population(self, country='IND'):
        """World Bank API"""
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/SP.POP.TOTL?format=json&date=2010:2024"
        try:
            data = requests.get(url, timeout=10).json()
            df = pd.DataFrame(data[1])
            print(f"✓ World Bank: {len(df)} records")
            return df
        except Exception as e:
            print(f"⚠ World Bank failed: {e}")
            return None
    
    def fetch_world_bank_health_expenditure(self, country='IND'):
        """Healthcare spending as % of GDP"""
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/SH.XPD.CHEX.GD.ZS?format=json&date=2010:2024"
        try:
            data = requests.get(url, timeout=10).json()
            df = pd.DataFrame(data[1])
            print(f"✓ Health Expenditure: {len(df)} records")
            return df
        except Exception as e:
            return None
    
    def run_full_refresh(self):
        """Refresh all data sources"""
        print("\n🔄 LIVE DATA REFRESH")
        print("="*50)
        
        results = {
            'who_diabetes': self.fetch_who_diabetes(),
            'wb_population': self.fetch_world_bank_population(),
            'wb_health_exp': self.fetch_world_bank_health_expenditure(),
            'fetched_at': datetime.now()
        }
        
        # Save snapshots
        for name, df in results.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(f"data/live_{name}.csv", index=False)
        
        print(f"\n✅ Refresh complete at {results['fetched_at']}")
        return results


if __name__ == "__main__":
    pipeline = LiveDataPipeline()
    pipeline.run_full_refresh()