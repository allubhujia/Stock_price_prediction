import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation.save_processed import load_processed_data
from src.models.train_regression import load_model
from src.models.evaluate import evaluate_model, plot_predictions, plot_feature_importance

def main():
    print("Starting model evaluation...")
    X_train, X_test, y_train, y_test = load_processed_data()
    model = load_model()
    results = evaluate_model(model, X_test, y_test)
    plot_predictions(results['y_test'], results['y_pred'])
    plot_feature_importance(model, top_n=15)
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
