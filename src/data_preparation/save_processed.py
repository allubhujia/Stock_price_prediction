import numpy as np
from src.config.settings import X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH

def save_processed_data(X_train, X_test, y_train, y_test):
    np.save(X_TRAIN_PATH, X_train)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)
    
    print(f"\nProcessed data saved:")
    print(f"X_train: {X_TRAIN_PATH}")
    print(f"X_test: {X_TEST_PATH}")
    print(f"y_train: {Y_TRAIN_PATH}")
    print(f"y_test: {Y_TEST_PATH}")

def load_processed_data():
    X_train = np.load(X_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    print(f"\nLoaded processed data:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from src.data_preparation.load_data import load_raw_data
    from src.data_preparation.features import create_technical_features, create_target
    from src.data_preparation.preprocess import prepare_features_and_target, split_data, normalize_data
    
    df = load_raw_data()
    df = create_technical_features(df)
    df = create_target(df)
    
    X, y, _ = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_norm, X_test_norm, _, _ = normalize_data(X_train, X_test)
    
    save_processed_data(X_train_norm, X_test_norm, y_train, y_test)