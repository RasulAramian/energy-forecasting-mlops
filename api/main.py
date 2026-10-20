import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse
from src.model import EnergyLSTM

app = FastAPI(title="Energy Forecasting API", version="1.0.0")

model = EnergyLSTM(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1)
scaler = None

try:
    model.load_state_dict(torch.load("models/energy_lstm_model.pth", map_location=torch.device("cpu")))
    model.eval()
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print(f"Warning: Artifacts not found or failed to load. Details: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if len(payload.sequence) != 24:
        raise HTTPException(status_code=400, detail="Sequence length must be exactly 24 items.")
    
    if scaler is None or model is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded properly.")
    
    try:
        input_array = np.array(payload.sequence, dtype=np.float32).reshape(-1, 1)
        scaled_input = scaler.transform(input_array)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            scaled_prediction = model(input_tensor)
            
        prediction_unscaled = scaler.inverse_transform(scaled_prediction.numpy())
        predicted_value = float(prediction_unscaled[0][0])
        
        return PredictionResponse(predicted_value=predicted_value)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
