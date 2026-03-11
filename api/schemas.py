from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):

    features: List[float]


class PredictionResponse(BaseModel):

    asd_probability: float
    risk_level: str