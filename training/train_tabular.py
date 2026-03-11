import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from models.classical_models import (
    build_random_forest,
    build_xgboost,
    build_svm,
    build_lightgbm
)

from utils.metrics import classification_metrics
from utils.model_utils import save_sklearn_model
from utils.logger import get_logger
from utils.config import PROCESSED_DATA_DIR

logger = get_logger("TRAIN_TABULAR")


def load_dataset():

    path = os.path.join(
        PROCESSED_DATA_DIR,
        "multimodal_dataset.csv"
    )

    df = pd.read_csv(path)

    X = df.drop(columns=["Class/ASD"])

    y = df["Class/ASD"]

    return X, y


def train_model(name, model):

    X, y = load_dataset()

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")

    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(
        y_test,
        preds,
        probs
    )

    logger.info(f"{name} metrics {metrics}")

    save_sklearn_model(model, name)

    return metrics


def run():

    models = {

        "random_forest": build_random_forest(),

        "xgboost": build_xgboost(),

        "svm": build_svm(),

        "lightgbm": build_lightgbm()
    }

    results = {}

    for name, model in models.items():

        logger.info(f"Training {name}")

        results[name] = train_model(name, model)

    return results


if __name__ == "__main__":

    run()