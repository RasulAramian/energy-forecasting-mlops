import os
import numpy as np
import torch
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.model import EnergyLSTM
from src.train import load_and_preprocess_data


def evaluate_model():
    model_path = "models/energy_lstm_model.pth"
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler artifacts not found. Please run training first.")

    print("Loading test data...")
    datasets, scaler = load_and_preprocess_data()
    X_test, y_test = datasets["test"]

    print("Loading model and scaler...")
    model = EnergyLSTM(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    print("Running evaluation...")
    with torch.no_grad():
        y_pred_scaled = model(X_test)

    # Inverse transform predictions and actual values back to original scale
    y_pred_np = y_pred_scaled.numpy()
    y_true_np = y_test.numpy()  # اصلاح شد: بدون unsqueeze تا ابعاد ۲بعدی حفظ شود

    y_pred_inv = scaler.inverse_transform(y_pred_np)
    y_true_inv = scaler.inverse_transform(y_true_np)

    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)

    print("\n--- Evaluation Results ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")


if __name__ == "__main__":
    evaluate_model()
