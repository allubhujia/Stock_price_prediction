import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preparation.save_processed import load_processed_data
from src.models.train_regression import load_model
from src.models.infer import predict_return, predict_single_sample

def main():
    print("Starting inference...")

    X_train, X_test, y_train, y_test = load_processed_data()
    model = load_model()

    predictions = predict_return(model, X_test)

    print("\nPredictions for test set:")
    print(f"Total predictions: {len(predictions)}")
    print("\nFirst 10 predictions:")
    for i in range(min(10, len(predictions))):
        print(
            f"Sample {i+1}: "
            f"Predicted return={predictions[i]:.6f}, "
            f"Actual return={y_test[i]:.6f}, "
            f"Diff={abs(predictions[i]-y_test[i]):.6f}"
        )

    print("\n" + "=" * 50)
    last_prediction = predict_single_sample(model, X_test[-1])
    print(f"Latest predicted return: {last_prediction:.6f}")
    print(f"Actual return: {y_test[-1]:.6f}")
    print(f"Difference: {abs(last_prediction - y_test[-1]):.6f}")
    print("=" * 50)

    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()
