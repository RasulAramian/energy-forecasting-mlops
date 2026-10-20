import torch
from torch.utils.data import Dataset
import numpy as np


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
