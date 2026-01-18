import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation.load_data import download_stock_data, save_raw_data, load_raw_data
from src.data_preparation.features import create_technical_features, create_target
from src.data_preparation.preprocess import prepare_features_and_target, split_data, normalize_data
from src.data_preparation.save_processed import save_processed_data

def main():
    print("Starting data preprocessing...")

    try:
        df = load_raw_data()
    except FileNotFoundError:
        df = download_stock_data()
        save_raw_data(df)

    df = create_technical_features(df)
    df = create_target(df)

    X, y, _ = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_norm, X_test_norm, _, _ = normalize_data(X_train, X_test)

    save_processed_data(X_train_norm, X_test_norm, y_train, y_test)

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
