import torch
from src.model import EnergyLSTM

def test_lstm_output_shape():
    # Initialize model with same parameters
    model = EnergyLSTM(input_dim=1, hidden_dim=32, num_layers=1, output_dim=1)
    model.eval()
    
    # Create a dummy batch: Batch size = 2, Sequence length = 30, Features = 1
    dummy_input = torch.randn(2, 30, 1)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Expected output shape: [Batch_Size, Output_Dim] -> [2, 1]
    assert output.shape == (2, 1), f"Expected shape (2, 1), but got {output.shape}"
