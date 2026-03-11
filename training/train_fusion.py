import joblib
import torch
import pandas as pd
import os

from models.fusion_model import FusionModel
from utils.config import PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("TRAIN_FUSION")


def run():

    tabular_model = joblib.load("models/saved/random_forest.joblib")

    motor_model = torch.load("models/saved/motor_lstm.pt")

    eye_model = torch.load("models/saved/eye_gru.pt")

    fusion = FusionModel(
        tabular_model,
        motor_model,
        eye_model
    )

    logger.info("Fusion model ready")

    return fusion