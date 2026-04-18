import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score

from utils.data_loader import load_split_dataset
from utils.logger import get_logger

logger = get_logger("PR_CURVE")


def plot_pr_curve(model_name, X_test, y_test):

    logger.info(f"Generating PR curve for {model_name}")

    model = joblib.load(f"models/saved/{model_name}.joblib")

    # Align features
    expected = getattr(model, "n_features_in_", None)
    X = X_test.values
    if expected is not None and X.shape[1] > expected:
        X = X[:, :expected]

    probs = model.predict_proba(X)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, probs)

    ap = average_precision_score(y_test, probs)

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


def run():

    _, X_test, _, y_test = load_split_dataset()

    # Convert to binary if multiclass
    if y_test.nunique() > 2:
        logger.warning("Multiclass detected - converting to binary")
        y_test = (y_test > 0).astype(int)

    models = [
        "random_forest",
        "xgboost",
        "svm",
        "lightgbm"
    ]

    for m in models:

        try:
            plot_pr_curve(m, X_test, y_test)

        except Exception as e:

            logger.error(f"{m} failed: {e}")


if __name__ == "__main__":

    run()
