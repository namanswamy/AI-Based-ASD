import os
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer

from utils.config import PROCESSED_DATA_DIR
from utils.metrics import classification_metrics
from utils.logger import get_logger

logger = get_logger("MODEL_EVALUATION")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data():

    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, "multimodal_dataset.csv")
    )

    logger.info(f"Loaded dataset {df.shape}")

    if "Class/ASD" not in df.columns:
        raise ValueError("Label column 'Class/ASD' not found")

    y = df["Class/ASD"]

    # Remove label and ID columns
    X = df.drop(
        columns=["Class/ASD", "participant_id"],
        errors="ignore"
    )

    return X, y


# --------------------------------------------------
# Evaluate one model
# --------------------------------------------------

def evaluate_model(model_name):

    model_path = f"models/saved/{model_name}.joblib"

    if not os.path.exists(model_path):
        logger.warning(f"{model_name} model not found")
        return None

    model = joblib.load(model_path)

    X, y = load_data()

    # -------------------------------
    # Handle missing values
    # -------------------------------

    imputer = SimpleImputer(strategy="mean")

    X = imputer.fit_transform(X)

    # -------------------------------
    # Align feature count
    # -------------------------------

    expected = getattr(model, "n_features_in_", None)

    if expected is not None:

        if X.shape[1] > expected:

            logger.warning(
                f"{model_name}: dropping extra columns "
                f"({X.shape[1]} -> {expected})"
            )

            X = X[:, :expected]

        elif X.shape[1] < expected:

            logger.warning(
                f"{model_name}: adding missing columns "
                f"({X.shape[1]} -> {expected})"
            )

            missing = expected - X.shape[1]

            X = np.hstack([X, np.zeros((X.shape[0], missing))])

    # -------------------------------
    # Predictions
    # -------------------------------

    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = preds

    metrics = classification_metrics(
        y,
        preds,
        probs
    )

    logger.info(f"{model_name} -> {metrics}")

    return metrics


# --------------------------------------------------
# Evaluate all models
# --------------------------------------------------

def run():

    models = [
        "random_forest",
        "xgboost",
        "svm",
        "lightgbm"
    ]

    results = {}

    for m in models:

        metrics = evaluate_model(m)

        if metrics is not None:
            results[m] = metrics

    return results


# --------------------------------------------------

if __name__ == "__main__":

    run()