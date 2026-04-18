from models.classical_models import (
    build_random_forest,
    build_xgboost,
    build_svm,
    build_lightgbm
)

from utils.data_loader import load_split_dataset
from utils.metrics import classification_metrics
from utils.model_utils import save_sklearn_model
from utils.logger import get_logger

logger = get_logger("TRAIN_TABULAR")


def train_model(name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, preds, probs)

    logger.info(f"{name} metrics {metrics}")

    save_sklearn_model(model, name)

    return metrics


def run():

    X_train, X_test, y_train, y_test = load_split_dataset()

    logger.info(
        f"Train: {X_train.shape}, Test: {X_test.shape}, "
        f"Features: {list(X_train.columns)}"
    )

    models = {
        "random_forest": build_random_forest(),
        "xgboost": build_xgboost(),
        "svm": build_svm(),
        "lightgbm": build_lightgbm()
    }

    results = {}

    for name, model in models.items():

        logger.info(f"Training {name}")

        results[name] = train_model(
            name, model, X_train, X_test, y_train, y_test
        )

    return results


if __name__ == "__main__":

    run()
