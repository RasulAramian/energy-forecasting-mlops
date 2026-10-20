from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    sequence: List[float] = Field(
        ..., 
        description="Sequence of 24 hourly energy consumption values",
        min_length=24,
        max_length=24
    )

class PredictionResponse(BaseModel):
    predicted_value: float
    status: str = "success"
