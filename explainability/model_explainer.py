import os
import shap
import joblib
import matplotlib.pyplot as plt

from utils.data_loader import load_split_dataset
from utils.logger import get_logger

logger = get_logger("MODEL_EXPLAINER")


class ModelExplainer:

    def __init__(self, model_path):

        self.model = joblib.load(model_path)

        logger.info(f"Loaded model {model_path}")

        # Use test set only for explanations
        _, X_test, _, self.y = load_split_dataset()

        # Align features
        expected = getattr(self.model, "n_features_in_", None)
        if expected is not None and X_test.shape[1] > expected:
            logger.warning(
                f"Dropping extra columns ({X_test.shape[1]} -> {expected})"
            )
            X_test = X_test.iloc[:, :expected]

        self.X = X_test

        self.explainer = shap.TreeExplainer(self.model)

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

        plt.savefig(save_path, bbox_inches="tight")

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

        plt.savefig(save_path, bbox_inches="tight")

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

        import pandas as pd
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

        plt.savefig(save_path, bbox_inches="tight")

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
