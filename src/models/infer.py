import numpy as np
from src.models.train_regression import load_model

def predict_return(model, X):
    predictions = model.predict(X)
    return predictions

def predict_single_sample(model, sample):
    sample = np.array(sample).reshape(1, -1)
    prediction = model.predict(sample)
    return prediction[0]

def predict_next_day_price(model, latest_features, last_close_price):
    predicted_return = predict_single_sample(model, latest_features)
    predicted_price = last_close_price * (1 + predicted_return)
    print(f"\nPredicted next day price: {predicted_price:.2f}")
    return predicted_price

if __name__ == "__main__":
    from src.data_preparation.save_processed import load_processed_data
    from src.data_preparation.load_data import load_raw_data

    X_train, X_test, y_train, y_test = load_processed_data()
    model = load_model()

    predictions = predict_return(model, X_test)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5]}")

    sample_return = predict_single_sample(model, X_test[0])
    print(f"\nSingle sample predicted return: {sample_return:.6f}")
    print(f"Actual return: {y_test[0]:.6f}")

    df = load_raw_data()
    last_close = df['Close'].iloc[-1]
    predict_next_day_price(model, X_test[-1], last_close)
