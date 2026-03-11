import shap
import joblib
import pandas as pd

from utils.config import PROCESSED_DATA_DIR


def run():

    df = pd.read_csv(
        f"{PROCESSED_DATA_DIR}/multimodal_dataset.csv"
    )

    X = df.drop(columns=["Class/ASD"])

    model = joblib.load("models/saved/best_model.joblib")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    shap.plots.bar(shap_values)

if __name__ == "__main__":

    run()