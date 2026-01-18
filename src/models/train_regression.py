from sklearn.ensemble import RandomForestRegressor
import joblib
from src.config.settings import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF, RF_RANDOM_STATE, MODEL_PATH
)

def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    print("Training completed!")
    return model

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")

def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"\nModel loaded from {path}")
    return model

if __name__ == "__main__":
    from src.data_preparation.save_processed import load_processed_data
    X_train, X_test, y_train, y_test = load_processed_data()
    model = train_random_forest(X_train, y_train)
    save_model(model)
