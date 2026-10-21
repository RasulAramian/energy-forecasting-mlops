import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest
from src.model import EnergyLSTM


model = None
scaler = None
is_model_loaded = False
is_scaler_loaded = False

MODEL_PATH = "models/energy_lstm_model.pth"
SCALER_PATH = "models/scaler.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, is_model_loaded, is_scaler_loaded

    try:
        if os.path.exists(MODEL_PATH):
            model = EnergyLSTM(
                input_dim=1,
                hidden_dim=50,
                num_layers=1,
                output_dim=1,
            )
            model.load_state_dict(
                torch.load(
                    MODEL_PATH,
                    map_location=torch.device("cpu"),
                )
            )
            model.eval()
            is_model_loaded = True
        else:
            is_model_loaded = False

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            is_scaler_loaded = True
        else:
            is_scaler_loaded = False

    except Exception as e:
        print(f"Error loading artifacts: {e}")
        is_model_loaded = False
        is_scaler_loaded = False

    yield


app = FastAPI(
    title="Energy Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    ready = is_model_loaded and is_scaler_loaded

    return {
        "status": "healthy" if ready else "not_ready",
        "model_loaded": is_model_loaded,
        "scaler_loaded": is_scaler_loaded,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    if not is_model_loaded or not is_scaler_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded properly.",
        )

    try:
        sequence = np.array(
            payload.sequence,
            dtype=np.float32,
        ).reshape(-1, 1)

        sequence_df = pd.DataFrame(
            sequence,
            columns=scaler.feature_names_in_,
        )
        scaled_sequence = scaler.transform(sequence_df)

        tensor_input = torch.tensor(
            scaled_sequence,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            prediction_scaled = model(tensor_input).item()

        prediction_array = np.array(
            [[prediction_scaled]],
            dtype=np.float32,
        )
        predicted_value = float(
            scaler.inverse_transform(prediction_array)[0][0]
        )

        return {
            "status": "success",
            "predicted_value": predicted_value,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}",
        )
