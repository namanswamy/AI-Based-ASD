import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.impute import SimpleImputer

from utils.config import PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("PR_CURVE")


# Load dataset

def load_data():

    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, "multimodal_dataset.csv")
    )

    logger.info(f"Loaded dataset {df.shape}")

    y = df["Class/ASD"]

    if y.nunique() > 2:
        logger.warning("Multiclass detected - converting to binary")
        y = (y > 0).astype(int)

    X = df.drop(
        columns=["Class/ASD", "participant_id"],
        errors="ignore"
    )

    return X, y


# Feature alignment

def align_features(model, X):

    expected = getattr(model, "n_features_in_", None)

    if expected is None:
        return X

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

        X = np.hstack([X.values, np.zeros((X.shape[0], missing))])

        return X

    return X


# Plot PR curve

def plot_pr_curve(model_name):

    logger.info(f"Generating PR curve for {model_name}")

    model = joblib.load(
        f"models/saved/{model_name}.joblib"
    )

    X, y = load_data()

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X = align_features(model, pd.DataFrame(X))

    probs = model.predict_proba(X)[:, 1]

    precision, recall, _ = precision_recall_curve(y, probs)

    ap = average_precision_score(y, probs)

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"{model_name} (AP={ap:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.legend()

    os.makedirs("outputs/plots", exist_ok=True)

    save_path = f"outputs/plots/pr_curve_{model_name}.png"

    plt.savefig(save_path)

    logger.info(f"Saved {save_path}")

    plt.close()


# Run for all models

def run():

    models = [
        "random_forest",
        "xgboost",
        "svm",
        "lightgbm"
    ]

    for m in models:

        try:
            plot_pr_curve(m)

        except Exception as e:

            logger.error(f"{m} failed: {e}")


if __name__ == "__main__":

    run()
