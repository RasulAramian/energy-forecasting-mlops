import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.model import EnergyLSTM


def load_and_preprocess_data(
    data_path="data/energy_consumption.csv",
    datetime_col="Datetime",
    feature_col="AEP_MW",
    seq_len=24
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)

    # Validate columns
    for col in [datetime_col, feature_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' was not found in CSV columns: {list(df.columns)}")

    # Sort chronologically and clean data
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    df = df.dropna(subset=[feature_col])

    values = df[[feature_col]].to_numpy(dtype=np.float32)

    # Time-series split (e.g., 70% train, 15% val, 15% test)
    train_size = int(len(values) * 0.70)
    val_size = int(len(values) * 0.15)

    train_data = values[:train_size]
    val_data = values[train_size - seq_len:train_size + val_size]
    test_data = values[train_size + val_size - seq_len:]

    # Fit MinMaxScaler ONLY on training data to avoid data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_val, y_val = create_sequences(val_scaled, seq_len)
    X_test, y_test = create_sequences(test_scaled, seq_len)

    return {
        "train": (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        "val": (torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        "test": (torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
    }, scaler


def train_model():
    os.makedirs("models", exist_ok=True)
    data_path = "data/energy_consumption.csv"

    print("Loading and preprocessing data...")
    datasets, scaler = load_and_preprocess_data(data_path)

    train_loader = DataLoader(
        TensorDataset(datasets["train"][0], datasets["train"][1]),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(datasets["val"][0], datasets["val"][1]),
        batch_size=32,
        shuffle=False
    )

    model = EnergyLSTM(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    model_path = "models/energy_lstm_model.pth"
    scaler_path = "models/scaler.pkl"

    print("Starting training...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            joblib.dump(scaler, scaler_path)
            print("--> Saved best model and scaler artifacts.")

    print("Training finished successfully!")


if __name__ == "__main__":
    train_model()
