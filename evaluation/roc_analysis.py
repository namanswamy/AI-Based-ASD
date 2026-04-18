import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from utils.data_loader import load_split_dataset
from utils.logger import get_logger

logger = get_logger("ROC_ANALYSIS")


def run():

    _, X_test, _, y_test = load_split_dataset()

    # Convert to binary if multiclass
    if y_test.nunique() > 2:
        logger.warning("Converting multiclass labels to binary")
        y_test = (y_test > 0).astype(int)

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

        # Align features
        expected = getattr(model, "n_features_in_", None)
        X = X_test.values
        if expected is not None and X.shape[1] > expected:
            X = X[:, :expected]

        probs = model.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)

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

    plt.close()


if __name__ == "__main__":
    run()
