import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation.save_processed import load_processed_data
from src.models.train_regression import train_random_forest, save_model

def main():
    print("Starting model training...")
    X_train, X_test, y_train, y_test = load_processed_data()
    model = train_random_forest(X_train, y_train)
    save_model(model)
    print("\nTraining completed successfully!")
if __name__ == "__main__":
    main()