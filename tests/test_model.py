import torch
from src.model import EnergyLSTM


def test_lstm_output_shape():
    model = EnergyLSTM(
        input_dim=1,
        hidden_dim=50,
        num_layers=1,
        output_dim=1
    )
    model.eval()

    dummy_input = torch.randn(2, 24, 1)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1), (
        f"Expected shape (2, 1), but got {output.shape}"
    )

    assert torch.isfinite(output).all(), (
        "Model output contains NaN or infinite values."
    )
