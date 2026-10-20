import os
import copy
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from src.dataset import EnergyDataset, create_sequences
from src.model import EnergyLSTM


def load_and_preprocess_data(
    file_path: str,
    datetime_col: str = "datetime",
    feature_col: str = "consumption_mw",
    seq_length: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    """Load, split, scale, and prepare time-series datasets."""

    if seq_length < 1:
        raise ValueError("seq_length must be at least 1.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)

    for col in [datetime_col, feature_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' was not found.")

    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        errors="raise"
    )

    df = df.sort_values(datetime_col).reset_index(drop=True)

    if df[datetime_col].duplicated().any():
        raise ValueError(
            "Duplicate timestamps must be handled before splitting."
        )

    values = df[[feature_col]].to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError(
            "The feature column contains missing or infinite values."
        )

    total_rows = len(values)

    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    if train_end <= seq_length:
        raise ValueError(
            "The training split is too small for the selected sequence length."
        )

    if val_end <= train_end:
        raise ValueError("The validation split is empty.")

    if val_end >= total_rows:
        raise ValueError("The test split is empty.")

    train_values = values[:train_end]

    scaler = MinMaxScaler()
    scaler.fit(train_values)

    scaled_values = scaler.transform(values).astype(np.float32)

    train_scaled = scaled_values[:train_end]

    val_context = scaled_values[
        train_end - seq_length:val_end
    ]

    test_context = scaled_values[
        val_end - seq_length:
    ]

    X_train, y_train = create_sequences(
        train_scaled,
        seq_length
    )

    X_val, y_val = create_sequences(
        val_context,
        seq_length
    )

    X_test, y_test = create_sequences(
        test_context,
        seq_length
    )

    datasets = {
        "train": EnergyDataset(X_train, y_train),
        "validation": EnergyDataset(X_val, y_val),
        "test": EnergyDataset(X_test, y_test)
    }

    print(f"Total rows: {total_rows}")
    print(f"Training samples: {len(datasets['train'])}")
    print(f"Validation samples: {len(datasets['validation'])}")
    print(f"Test samples: {len(datasets['test'])}")

    return datasets, scaler


def train_model(
    data_path: str = "data/energy_consumption.csv",
    models_dir: str = "models",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42
):
    """Train EnergyLSTM and save the best model and fitted scaler."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be a positive finite number.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(models_dir, exist_ok=True)

    datasets, scaler = load_and_preprocess_data(data_path)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        shuffle=False
    )

    model = EnergyLSTM(
        input_dim=1,
        hidden_dim=50,
        num_layers=1,
        output_dim=1
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0

        for sequences, targets in train_loader:
            optimizer.zero_grad()

            outputs = model(sequences)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * len(sequences)

        train_loss /= len(datasets["train"])

        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for sequences, targets in val_loader:
                outputs = model(sequences)

                loss = criterion(outputs, targets)

                val_loss += loss.item() * len(sequences)

        val_loss /= len(datasets["validation"])

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"- Train Loss: {train_loss:.6f} "
            f"- Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_state_dict = copy.deepcopy(
                model.state_dict()
            )

    if best_state_dict is None:
        raise RuntimeError(
            "Training did not produce a valid model checkpoint."
        )

    model.load_state_dict(best_state_dict)

    model_path = os.path.join(
        models_dir,
        "energy_lstm_model.pth"
    )

    scaler_path = os.path.join(
        models_dir,
        "scaler.pkl"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    return model, scaler, datasets


if __name__ == "__main__":
    train_model()
