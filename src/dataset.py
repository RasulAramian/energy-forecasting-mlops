import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class EnergyDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def prepare_data(
    df: pd.DataFrame,
    feature_col: str,
    seq_length: int = 24,
    scaler: MinMaxScaler = None
):
    if feature_col not in df.columns:
        raise ValueError(f"Column '{feature_col}' does not exist.")

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    if seq_length < 1:
        raise ValueError("seq_length must be at least 1.")

    values = df[[feature_col]].to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError("The feature column contains missing or infinite values.")

    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(values)
    else:
        scaled_data = scaler.transform(values)

    if len(scaled_data) <= seq_length:
        raise ValueError("Not enough data to create sequences.")

    sequences = []
    targets = []

    for i in range(len(scaled_data) - seq_length):
        seq = scaled_data[i:i + seq_length]
        target = scaled_data[i + seq_length]

        sequences.append(seq)
        targets.append(target)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        scaler
    )
