import pandas as pd
import joblib
import matplotlib.pyplot as plt


def run():

    model = joblib.load("models/saved/best_model.joblib")

    importance = model.feature_importances_

    features = pd.read_csv(
        "data/processed/multimodal_dataset.csv"
    ).drop(columns=["Class/ASD"]).columns

    df = pd.DataFrame({

        "feature": features,

        "importance": importance

    })

    df = df.sort_values(
        by="importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))

    plt.barh(df.feature[:15], df.importance[:15])

    plt.title("Top Features")

    plt.savefig("outputs/plots/feature_importance.png")

    plt.show()


if __name__ == "__main__":

    run()