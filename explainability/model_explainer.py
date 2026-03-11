import os
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from utils.config import PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("MODEL_EXPLAINER")


class ModelExplainer:

    def __init__(self, model_path):

        self.model = joblib.load(model_path)

        logger.info(f"Loaded model {model_path}")

        self.data = self.load_data()

        self.X = self.data.drop(
            columns=["Class/ASD", "participant_id"],
            errors="ignore"
        )

        self.y = self.data["Class/ASD"]

        if hasattr(self.model, "n_features_in_"):

            expected = self.model.n_features_in_

            if self.X.shape[1] > expected:
                logger.warning(
                    f"Dropping extra columns ({self.X.shape[1]} -> {expected})"
                )
                self.X = self.X.iloc[:, :expected]

        self.explainer = shap.TreeExplainer(self.model)

    # Load dataset

    def load_data(self):

        path = os.path.join(
            PROCESSED_DATA_DIR,
            "multimodal_dataset.csv"
        )

        df = pd.read_csv(path)

        logger.info(f"Loaded dataset {df.shape}")

        return df

    # Global SHAP explanation

    def shap_summary(self):

        logger.info("Generating SHAP summary")

        shap_values = self.explainer.shap_values(self.X)

        shap.summary_plot(
            shap_values,
            self.X,
            show=False
        )

        os.makedirs("outputs/plots", exist_ok=True)

        save_path = "outputs/plots/shap_summary.png"

        plt.savefig(save_path)

        logger.info(f"Saved {save_path}")

        plt.close()

    # SHAP Feature Importance

    def shap_bar(self):

        logger.info("Generating SHAP feature importance")

        shap_values = self.explainer.shap_values(self.X)

        shap.summary_plot(
            shap_values,
            self.X,
            plot_type="bar",
            show=False
        )

        os.makedirs("outputs/plots", exist_ok=True)

        save_path = "outputs/plots/shap_bar.png"

        plt.savefig(save_path)

        logger.info(f"Saved {save_path}")

        plt.close()

    # Local explanation for a sample

    def explain_instance(self, index):

        logger.info(f"Explaining instance {index}")

        sample = self.X.iloc[index:index+1]

        shap_values = self.explainer.shap_values(sample)

        shap.force_plot(
            self.explainer.expected_value,
            shap_values,
            sample
        )

    # Feature importance using model

    def feature_importance(self):

        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Model does not support feature importance")
            return

        importances = self.model.feature_importances_

        features = self.X.columns

        df = pd.DataFrame({
            "feature": features,
            "importance": importances
        })

        df = df.sort_values(
            by="importance",
            ascending=False
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            df.feature[:15],
            df.importance[:15]
        )

        plt.title("Feature Importance")

        os.makedirs("outputs/plots", exist_ok=True)

        save_path = "outputs/plots/model_feature_importance.png"

        plt.savefig(save_path)

        logger.info(f"Saved {save_path}")

        plt.close()


def run():

    model_path = "models/saved/random_forest.joblib"

    explainer = ModelExplainer(model_path)

    explainer.shap_summary()

    explainer.shap_bar()

    explainer.feature_importance()


if __name__ == "__main__":

    run()
