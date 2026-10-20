import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class EnergyDataset(Dataset):
    """
    Custom PyTorch Dataset for time-series forecasting.
    """
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_and_preprocess_data(file_path_or_url: str, seq_length: int = 30):
    """
    Loads time-series data, normalizes it, and creates rolling windows.
    """
    df = pd.read_csv(file_path_or_url)
    df.columns = ['date', 'consumption']
    
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[['consumption']])
    
    sequences = []
    targets = []
    
    for i in range(len(scaled_values) - seq_length):
        seq = scaled_values[i:i + seq_length]
        target = scaled_values[i + seq_length]
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(targets), scaler
