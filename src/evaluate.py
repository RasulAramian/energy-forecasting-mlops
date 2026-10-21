import os

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.model import EnergyLSTM
from src.train import load_and_preprocess_data


def evaluate_model():
    model_path = "models/energy_lstm_model.pth"
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Please run training first."
        )

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_path}. Please run training first."
        )

    print("Loading test data...")
    datasets, _ = load_and_preprocess_data()
    X_test, y_test = datasets["test"]

    print("Loading model and saved scaler...")
    scaler = joblib.load(scaler_path)

    model = EnergyLSTM(
        input_dim=1,
        hidden_dim=50,
        num_layers=1,
        output_dim=1,
    )

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()

    print("Running evaluation...")
    with torch.no_grad():
        y_pred_scaled = model(X_test)

    y_pred_np = y_pred_scaled.numpy()
    y_true_np = y_test.numpy()

    y_pred_inv = scaler.inverse_transform(y_pred_np)
    y_true_inv = scaler.inverse_transform(y_true_np)

    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mae = mean_absolute_error(y_true_inv, y_pred_inv)

    print("\n--- Evaluation Results ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} MW")
    print(f"Mean Absolute Error (MAE): {mae:.4f} MW")


if __name__ == "__main__":
    evaluate_model()
