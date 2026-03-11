import pandas as pd
import os

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("MOTOR_PREPROCESSOR")


class MotorFeatureExtractor:

    def load(self):

        path = os.path.join(RAW_DATA_DIR, "motor_pattern.csv")

        df = pd.read_csv(path)

        logger.info(f"Loaded motor dataset {df.shape}")

        return df

    def process(self):

        df = self.load()

        numeric = df.select_dtypes(include="number")

        if "label" not in numeric.columns:

            numeric["label"] = 0

        output = os.path.join(
            PROCESSED_DATA_DIR,
            "motor_features.csv"
        )

        numeric.to_csv(output, index=False)

        logger.info(f"Saved motor features {numeric.shape}")


def run():

    MotorFeatureExtractor().process()


if __name__ == "__main__":

    run()
