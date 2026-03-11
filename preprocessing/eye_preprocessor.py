import pandas as pd
import numpy as np
import os

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("EYE_PREPROCESSOR")


class EyeFeatureExtractor:

    def load(self):

        path = os.path.join(RAW_DATA_DIR, "eye_tracking.csv")

        df = pd.read_csv(path)

        logger.info(f"Loaded eye tracking {df.shape}")

        logger.info(f"Columns detected: {list(df.columns)}")

        return df

    # --------------------------------------------------
    # Feature extraction
    # --------------------------------------------------

    def extract_features(self, df):

        features = {}

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in eye tracking data")

        # Statistical features
        for col in numeric_cols:

            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_var"] = df[col].var()
            features[f"{col}_max"] = df[col].max()
            features[f"{col}_min"] = df[col].min()

        # Motion magnitude if coordinates exist
        if len(numeric_cols) >= 2:

            diff = np.diff(df[numeric_cols].values, axis=0)

            motion = np.linalg.norm(diff, axis=1)

            features["motion_mean"] = np.mean(motion)
            features["motion_std"] = np.std(motion)

        # Blink rate if blink column exists
        if "blink" in df.columns:

            features["blink_rate"] = df["blink"].sum() / len(df)

        # Convert to DataFrame
        features_df = pd.DataFrame([features])

        # --------------------------------------------------
        # REQUIRED COLUMNS FOR TRAINING PIPELINE
        # --------------------------------------------------

        features_df["participant_id"] = 0

        # placeholder label (will be replaced by merge later)
        features_df["Class/ASD"] = 0

        return features_df

    # --------------------------------------------------

    def process(self):

        df = self.load()

        features = self.extract_features(df)

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        output = os.path.join(
            PROCESSED_DATA_DIR,
            "eye_features.csv"
        )

        features.to_csv(output, index=False)

        logger.info(f"Saved eye features -> {output}")


def run():

    EyeFeatureExtractor().process()


if __name__ == "__main__":

    run()