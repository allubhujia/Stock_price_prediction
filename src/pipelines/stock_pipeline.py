from src.data_preparation.load_data import download_stock_data
from src.data_preparation.features import create_technical_features, create_target
from src.data_preparation.preprocess import prepare_features_and_target, split_data, normalize_data
from src.data_preparation.save_processed import save_processed_data
from src.models.train_regression import train_random_forest, save_model
from src.models.evaluate import (
    evaluate_model,
    plot_returns_timeseries,
    plot_returns_scatter,
    plot_feature_importance
)

def run_full_pipeline():
    print("=" * 70)
    print("STOCK RETURN PREDICTION PIPELINE - RANDOM FOREST")
    print("=" * 70)

    print("\n[STEP 1/6] Loading stock data...")
    df = download_stock_data()

    print("\n[STEP 2/6] Creating technical features...")
    df = create_technical_features(df)
    df = create_target(df)

    print("\n[STEP 3/6] Preparing and splitting data...")
    X, y, feature_cols = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_norm, X_test_norm, _, _ = normalize_data(X_train, X_test)

    print("\n[STEP 4/6] Saving processed data...")
    save_processed_data(X_train_norm, X_test_norm, y_train, y_test)

    print("\n[STEP 5/6] Training Random Forest model...")
    model = train_random_forest(X_train_norm, y_train)
    save_model(model)

    print("\n[STEP 6/6] Evaluating model performance...")
    results = evaluate_model(model, X_test_norm, y_test)

    plot_returns_timeseries(results['y_test'], results['y_pred'])
    plot_returns_scatter(results['y_test'], results['y_pred'])
    plot_feature_importance(model, feature_names=feature_cols, top_n=15)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return model, results

if __name__ == "__main__":
    model, results = run_full_pipeline()
