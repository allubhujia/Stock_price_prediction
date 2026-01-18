import pandas as pd
import numpy as np
from src.config.settings import WINDOW_SIZES

def create_technical_features(df):
    df = df.copy()

    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    for window in WINDOW_SIZES:
        df[f'MA_{window}'] = df['Close'].rolling(window).mean()
        df[f'STD_{window}'] = df['Close'].rolling(window).std()
        df[f'Volume_MA_{window}'] = df['Total Trade Quantity'].rolling(window).mean()

    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

    df['Volatility_5'] = df['return'].rolling(5).std()
    df['Volatility_10'] = df['return'].rolling(10).std()

    df['HL_Spread'] = df['High'] - df['Low']
    df['HL_Pct'] = (df['High'] - df['Low']) / df['Close']

    df['VWAP'] = (
        (df['Total Trade Quantity'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
        / df['Total Trade Quantity'].cumsum()
    )

    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Total Trade Quantity'].shift(lag)

    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter

    return df

def create_target(df):
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    from src.data_preparation.load_data import load_raw_data

    df = load_raw_data()
    df = create_technical_features(df)
    df = create_target(df)
    print(df.columns.tolist())
    print(df.shape)
