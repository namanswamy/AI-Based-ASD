import os
import joblib
import numpy as np

from utils.data_loader import load_split_dataset
from utils.metrics import classification_metrics
from utils.logger import get_logger

logger = get_logger("MODEL_EVALUATION")


def evaluate_model(model_name, X_test, y_test):

    model_path = f"models/saved/{model_name}.joblib"

    if not os.path.exists(model_path):
        logger.warning(f"{model_name} model not found")
        return None

    model = joblib.load(model_path)

    # Align feature count if needed
    expected = getattr(model, "n_features_in_", None)

    X = X_test.values if hasattr(X_test, "values") else X_test

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

    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = preds

    metrics = classification_metrics(y_test, preds, probs)

    logger.info(f"{model_name} -> {metrics}")

    return metrics


def run():

    _, X_test, _, y_test = load_split_dataset()

    models = [
        "random_forest",
        "xgboost",
        "svm",
        "lightgbm"
    ]

    results = {}

    for m in models:

        metrics = evaluate_model(m, X_test, y_test)

        if metrics is not None:
            results[m] = metrics

    return results


if __name__ == "__main__":

    run()
