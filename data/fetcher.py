import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


class FREDDataFetcher:
    """Fetches and caches market data from FRED API."""
    
    def __init__(self, cache_dir='./data/cache', api_key=None):
        self.cache_dir = cache_dir
        self.api_key = api_key
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_series(self, series_id: str, start_date: str = None) -> pd.Series:
        """Fetch FRED series with local caching."""
        cache_path = os.path.join(self.cache_dir, f"{series_id}.pkl")
        
        # Check cache freshness (< 1 day old)
        if os.path.exists(cache_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - mod_time < timedelta(days=1):
                print(f"Loading {series_id} from cache...")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Fetch from FRED
        print(f"Fetching {series_id} from FRED...")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        try:
            series = pdr.DataReader(series_id, 'fred', start_date)
            
            # Cache result
            with open(cache_path, 'wb') as f:
                pickle.dump(series, f)
            
            return series
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            # Return dummy data for development
            dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
            return pd.Series([0.05] * len(dates), index=dates, name=series_id)
    
    def build_yield_curve(self, as_of_date: str = None) -> pd.DataFrame:
        """Construct yield curve from Treasury rates."""
        tenors = {
            'DGS1': 1,
            'DGS2': 2,
            'DGS5': 5,
            'DGS10': 10,
            'DGS30': 30
        }
        
        curves = {}
        for series_id, years in tenors.items():
            try:
                series = self.fetch_series(series_id)
                if as_of_date:
                    value = series.loc[as_of_date].iloc[0] if as_of_date in series.index else series.iloc[-1].iloc[0]
                else:
                    value = series.iloc[-1].iloc[0]
                curves[years] = value / 100.0  # Convert percentage to decimal
            except Exception as e:
                print(f"Warning: Using fallback for {series_id}")
                # Fallback values
                curves[years] = 0.03 + 0.01 * (years / 10)
        
        df = pd.DataFrame({
            'tenor': list(curves.keys()),
            'rate': list(curves.values())
        }).sort_values('tenor')
        
        return df
    
    def fetch_credit_spread(self, rating='IG', as_of_date=None) -> float:
        """Fetch OAS for credit curve bootstrapping."""
        series_map = {
            'IG': 'BAMLC0A0CM',
            'AAA': 'BAMLC0A1CAAA'
        }
        
        try:
            series = self.fetch_series(series_map.get(rating, 'BAMLC0A0CM'))
            if as_of_date:
                return series.loc[as_of_date].iloc[0] / 10000.0  # bps to decimal
            return series.iloc[-1].iloc[0] / 10000.0
        except:
            print(f"Warning: Using fallback credit spread")
            return 0.015  # 150 bps fallback
    
    def fetch_sofr_history(self, lookback_days=252) -> pd.Series:
        """Fetch SOFR historical data for calibration."""
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        return self.fetch_series('SOFR', start_date)


if __name__ == "__main__":
    # Test the fetcher
    fetcher = FREDDataFetcher()
    
    print("Testing yield curve construction...")
    yc = fetcher.build_yield_curve()
    print(yc)
    
    print("\nTesting credit spread fetch...")
    oas = fetcher.fetch_credit_spread('IG')
    print(f"IG OAS: {oas*10000:.2f} bps")