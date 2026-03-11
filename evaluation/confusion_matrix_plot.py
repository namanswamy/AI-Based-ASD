import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix
from utils.config import PROCESSED_DATA_DIR


def run():

    df = pd.read_csv(
        f"{PROCESSED_DATA_DIR}/multimodal_dataset.csv"
    )

    X = df.drop(columns=["Class/ASD"])

    y = df["Class/ASD"]

    model = joblib.load("models/saved/best_model.joblib")

    preds = model.predict(X)

    cm = confusion_matrix(y, preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.savefig("outputs/plots/confusion_matrix.png")

    plt.show()


if __name__ == "__main__":

    run()