import joblib
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("INFERENCE")


class ASDInferenceEngine:

    def __init__(self):

        self.model = joblib.load(
            "models/saved/random_forest.joblib"
        )

        logger.info("Model loaded")

    def predict(self, features):

        X = np.array(features).reshape(1, -1)

        prob = self.model.predict_proba(X)[0][1]

        risk = self.risk_level(prob)

        return prob, risk

    def risk_level(self, prob):

        if prob < 0.3:
            return "Low Risk"

        elif prob < 0.6:
            return "Moderate Risk"

        else:
            return "High Risk"