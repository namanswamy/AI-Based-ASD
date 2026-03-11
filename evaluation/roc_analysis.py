import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer

from utils.config import PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("ROC_ANALYSIS")


# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

def load_data():

    path = os.path.join(
        PROCESSED_DATA_DIR,
        "multimodal_dataset.csv"
    )

    df = pd.read_csv(path)

    logger.info(f"Loaded dataset {df.shape}")

    # Convert labels to binary (ASD vs Non-ASD)
    y = df["Class/ASD"]

    if y.nunique() > 2:
        logger.warning("Converting multiclass labels to binary")
        y = (y > 0).astype(int)

    X = df.drop(
        columns=["Class/ASD", "participant_id"],
        errors="ignore"
    )

    return X, y


# ---------------------------------------------------
# Align features with trained model
# ---------------------------------------------------

def align_features(model, X):

    expected = getattr(model, "n_features_in_", None)

    if expected is None:
        return X.values

    if X.shape[1] > expected:
        logger.warning(
            f"Dropping extra columns ({X.shape[1]} -> {expected})"
        )
        X = X.iloc[:, :expected]

    elif X.shape[1] < expected:
        logger.warning(
            f"Adding missing columns ({X.shape[1]} -> {expected})"
        )

        missing = expected - X.shape[1]

        for i in range(missing):
            X[f"missing_{i}"] = 0

    return X.values


# ---------------------------------------------------
# ROC plotting
# ---------------------------------------------------

def run():

    X, y = load_data()

    # Handle NaN values
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X))

    models = [
        "random_forest",
        "xgboost",
        "lightgbm"
    ]

    plt.figure(figsize=(8, 6))

    for m in models:

        model_path = f"models/saved/{m}.joblib"

        if not os.path.exists(model_path):
            logger.warning(f"{m} model not found")
            continue

        model = joblib.load(model_path)

        X_aligned = align_features(model, X)

        probs = model.predict_proba(X_aligned)[:, 1]

        fpr, tpr, _ = roc_curve(y, probs)

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{m} (AUC={roc_auc:.3f})"
        )

        logger.info(f"{m} AUC = {roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve Comparison")

    plt.legend()

    os.makedirs("outputs/plots", exist_ok=True)

    save_path = "outputs/plots/roc_curve.png"

    plt.savefig(save_path)

    logger.info(f"Saved ROC curve -> {save_path}")

    plt.show()


if __name__ == "__main__":
    run()