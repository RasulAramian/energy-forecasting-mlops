import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class EnergyDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def prepare_data(df: pd.DataFrame, feature_col: str, seq_length: int = 24):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[[feature_col]])
    
    sequences = []
    targets = []
    
    for i in range(len(scaled_data) - seq_length):
        seq = scaled_data[i:i + seq_length]
        target = scaled_data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(targets), scaler
