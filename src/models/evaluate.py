import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("Model Performance Metrics:")
    print("="*50)
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50 + "\n")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test
    }

def plot_returns_timeseries(y_test, y_pred, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual Returns', alpha=0.7)
    plt.plot(y_pred, label='Predicted Returns', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Return')
    plt.title('Actual vs Predicted Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_returns_scatter(y_test, y_pred, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2)
    plt.xlabel('Actual Return')
    plt.ylabel('Predicted Return')
    plt.title('Return Prediction Scatter')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_feature_importance(model, feature_names=None, top_n=15, save_path=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))

    labels = (
        [feature_names[i] for i in indices]
        if feature_names is not None
        else [f'Feature {i}' for i in indices]
    )

    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), labels)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    from src.data_preparation.save_processed import load_processed_data
    from src.models.train_regression import load_model

    X_train, X_test, y_train, y_test = load_processed_data()
    model = load_model()

    results = evaluate_model(model, X_test, y_test)
    plot_returns_timeseries(results['y_test'], results['y_pred'])
    plot_returns_scatter(results['y_test'], results['y_pred'])
    plot_feature_importance(model, top_n=15)
