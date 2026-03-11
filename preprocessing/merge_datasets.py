import pandas as pd
import os

from utils.config import PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("MERGE_DATASETS")


class DatasetMerger:

    def process(self):

        tabular = pd.read_csv(
            os.path.join(PROCESSED_DATA_DIR, "tabular_features.csv")
        )

        eye = pd.read_csv(
            os.path.join(PROCESSED_DATA_DIR, "eye_features.csv")
        )

        motor = pd.read_csv(
            os.path.join(PROCESSED_DATA_DIR, "motor_features.csv")
        )

        clinical = pd.read_csv(
            os.path.join(PROCESSED_DATA_DIR, "clinical_features.csv")
        )

        merged = pd.concat(
            [tabular, eye, motor, clinical],
            axis=1
        )

        output = os.path.join(
            PROCESSED_DATA_DIR,
            "multimodal_dataset.csv"
        )

        merged.to_csv(output, index=False)

        logger.info(f"Saved {output}")


def run():

    DatasetMerger().process()


if __name__ == "__main__":

    run()