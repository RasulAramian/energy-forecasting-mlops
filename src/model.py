import torch
import torch.nn as nn

class EnergyLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=1, output_dim=1):
        super(EnergyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out
