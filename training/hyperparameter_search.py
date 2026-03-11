import os
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.metrics import make_scorer, f1_score

from models.classical_models import (
    build_random_forest,
    build_xgboost,
    build_lightgbm
)

from utils.config import PROCESSED_DATA_DIR
from utils.logger import get_logger
from utils.model_utils import save_sklearn_model
from utils.metrics import classification_metrics
from utils.experiment_tracker import (
    start_experiment,
    log_metric,
    log_param,
    end_experiment
)

logger = get_logger("HYPERPARAM_SEARCH")


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_dataset():

    path = os.path.join(
        PROCESSED_DATA_DIR,
        "multimodal_dataset.csv"
    )

    df = pd.read_csv(path)

    X = df.drop(columns=["Class/ASD"])

    y = df["Class/ASD"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ---------------------------------------------------------
# Random Forest search
# ---------------------------------------------------------

def tune_random_forest(X_train, y_train):

    model = build_random_forest()

    param_grid = {

        "n_estimators": [100, 200, 300, 400],

        "max_depth": [6, 8, 10, 12],

        "min_samples_split": [2, 5, 10],

        "min_samples_leaf": [1, 2, 4]
    }

    search = GridSearchCV(

        model,

        param_grid,

        scoring="f1",

        cv=5,

        n_jobs=-1,

        verbose=2
    )

    search.fit(X_train, y_train)

    logger.info(f"Best RF params {search.best_params_}")

    return search.best_estimator_


# ---------------------------------------------------------
# XGBoost search
# ---------------------------------------------------------

def tune_xgboost(X_train, y_train):

    model = build_xgboost()

    param_grid = {

        "n_estimators": [200, 300, 400],

        "max_depth": [4, 6, 8],

        "learning_rate": [0.01, 0.05, 0.1],

        "subsample": [0.8, 0.9, 1.0]
    }

    search = RandomizedSearchCV(

        model,

        param_grid,

        n_iter=10,

        scoring="f1",

        cv=5,

        n_jobs=-1,

        verbose=2
    )

    search.fit(X_train, y_train)

    logger.info(f"Best XGB params {search.best_params_}")

    return search.best_estimator_


# ---------------------------------------------------------
# LightGBM search
# ---------------------------------------------------------

def tune_lightgbm(X_train, y_train):

    model = build_lightgbm()

    param_grid = {

        "n_estimators": [200, 300, 400],

        "num_leaves": [31, 63, 127],

        "learning_rate": [0.01, 0.05, 0.1],

        "max_depth": [-1, 10, 20]
    }

    search = RandomizedSearchCV(

        model,

        param_grid,

        n_iter=10,

        scoring="f1",

        cv=5,

        n_jobs=-1,

        verbose=2
    )

    search.fit(X_train, y_train)

    logger.info(f"Best LGBM params {search.best_params_}")

    return search.best_estimator_


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(

        y_test,

        preds,

        probs
    )

    return metrics


# ---------------------------------------------------------
# Run search
# ---------------------------------------------------------

def run():

    logger.info("Starting hyperparameter search")

    start_experiment("hyperparameter_search")

    X_train, X_test, y_train, y_test = load_dataset()

    models = {

        "random_forest": tune_random_forest,

        "xgboost": tune_xgboost,

        "lightgbm": tune_lightgbm
    }

    results = {}

    best_score = 0
    best_model = None
    best_name = None

    for name, func in models.items():

        logger.info(f"Tuning {name}")

        model = func(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        results[name] = metrics

        log_metric(f"{name}_f1", metrics["f1_score"])

        if metrics["f1_score"] > best_score:

            best_score = metrics["f1_score"]

            best_model = model

            best_name = name

    logger.info(f"Best model {best_name}")

    save_sklearn_model(best_model, "best_model")

    log_param("best_model", best_name)

    end_experiment()

    return results


if __name__ == "__main__":

    run()