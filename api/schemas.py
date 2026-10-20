from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    sequence: List[float] = Field(
        ..., 
        description="Sequence of 24 hourly energy consumption values",
        min_items=24,
        max_items=24
    )

class PredictionResponse(BaseModel):
    predicted_value: float
    status: str = "success"
