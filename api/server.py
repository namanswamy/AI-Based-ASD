from fastapi import FastAPI
from api.schemas import PredictionRequest, PredictionResponse
from api.inference import ASDInferenceEngine

from utils.logger import get_logger

logger = get_logger("API_SERVER")

app = FastAPI(
    title="ASD AI Research API",
    version="1.0"
)

engine = ASDInferenceEngine()


@app.get("/")
def home():

    return {

        "message": "ASD AI Research System",

        "status": "running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    prob, risk = engine.predict(data.features)

    return {

        "asd_probability": prob,

        "risk_level": risk
    }