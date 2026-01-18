import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config.settings import TEST_SIZE, RANDOM_STATE

def prepare_features_and_target(df):
    df = df.dropna().reset_index(drop=True)

    exclude_cols = [
        'Date',
        'Open',
        'High',
        'Low',
        'Close',
        'Total Trade Quantity'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols and col != 'return']

    X = df[feature_cols].values
    y = df['return'].values

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_cols}")

    return X, y, feature_cols


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, mean, std


if __name__ == "__main__":
    from src.data_preparation.load_data import load_raw_data
    from src.data_preparation.features import create_technical_features, create_target

    df = load_raw_data()
    df = create_technical_features(df)
    df = create_target(df)

    X, y, feature_cols = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)

    print(f"\nNormalized training set shape: {X_train_norm.shape}")
    print(f"Normalized testing set shape: {X_test_norm.shape}")
