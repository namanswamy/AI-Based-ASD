import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix

from utils.data_loader import load_split_dataset
from utils.logger import get_logger

logger = get_logger("CONFUSION_MATRIX")


def run():

    _, X_test, _, y_test = load_split_dataset()

    model = joblib.load("models/saved/best_model.joblib")

    # Align features
    expected = getattr(model, "n_features_in_", None)
    X = X_test.values
    if expected is not None and X.shape[1] > expected:
        X = X[:, :expected]

    preds = model.predict(X)

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    os.makedirs("outputs/plots", exist_ok=True)

    plt.savefig("outputs/plots/confusion_matrix.png")

    logger.info("Saved confusion matrix plot")

    plt.close()


if __name__ == "__main__":

    run()
