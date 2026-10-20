import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class EnergyDataset(Dataset):
    """PyTorch Dataset for time-series forecasting."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(values: np.ndarray, seq_length: int = 24):
    """Create rolling input sequences and next-step targets."""

    if seq_length < 1:
        raise ValueError("seq_length must be at least 1.")

    if len(values) <= seq_length:
        raise ValueError("Not enough data to create sequences.")

    sequences = []
    targets = []

    for i in range(len(values) - seq_length):
        sequences.append(values[i:i + seq_length])
        targets.append(values[i + seq_length])

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32)
    )


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

    df = pd.read_csv(file_path)

    required_columns = [datetime_col, feature_col]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' was not found.")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="raise")
    df = df.sort_values(datetime_col).reset_index(drop=True)

    if df[datetime_col].duplicated().any():
        raise ValueError("Duplicate timestamps must be handled before splitting.")

    values = df[[feature_col]].to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError("The consumption column contains missing or infinite values.")

    total_rows = len(values)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    if train_end <= seq_length:
        raise ValueError("The training split is too small for the selected sequence length.")

    train_values = values[:train_end]
    val_values = values[train_end:val_end]
    test_values = values[val_end:]

    scaler = MinMaxScaler()
    scaler.fit(train_values)

    scaled_values = scaler.transform(values).astype(np.float32)

    train_scaled = scaled_values[:train_end]
    val_scaled = scaled_values[train_end:val_end]
    test_scaled = scaled_values[val_end:]

    val_context = scaled_values[train_end - seq_length:val_end]
    test_context = scaled_values[val_end - seq_length:]

    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_val, y_val = create_sequences(val_context, seq_length)
    X_test, y_test = create_sequences(test_context, seq_length)

    datasets = {
        "train": EnergyDataset(X_train, y_train),
        "validation": EnergyDataset(X_val, y_val),
        "test": EnergyDataset(X_test, y_test)
    }

    return datasets, scaler
