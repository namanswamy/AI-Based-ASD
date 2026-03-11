from models.classical_models import (
    build_random_forest,
    build_svm,
    build_xgboost,
    build_lightgbm
)

from models.lstm_models import LSTMClassifier
from models.gru_models import GRUClassifier


MODEL_REGISTRY = {

    "random_forest": build_random_forest,

    "svm": build_svm,

    "xgboost": build_xgboost,

    "lightgbm": build_lightgbm,

    "lstm": LSTMClassifier,

    "gru": GRUClassifier
}


def get_model(name):

    if name not in MODEL_REGISTRY:

        raise ValueError(f"Model {name} not found")

    return MODEL_REGISTRY[name]