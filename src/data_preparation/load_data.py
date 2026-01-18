import pandas as pd
from src.config.settings import RAW_DATA_PATH

def download_stock_data():
    print("Using existing stock data...")
    df = pd.read_csv(RAW_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(df)} records")
    return df

def save_raw_data(df, path=RAW_DATA_PATH):
    df.to_csv(path, index=False)
    print(f"Raw data saved to {path}")

def load_raw_data(path=RAW_DATA_PATH):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(df)} records")
    return df

if __name__ == "__main__":
    df = download_stock_data()
    save_raw_data(df)
