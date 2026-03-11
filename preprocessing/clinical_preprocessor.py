import os
import pandas as pd

from sklearn.preprocessing import StandardScaler

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger
from utils.validators import validate_dataframe

logger = get_logger("CLINICAL_PREPROCESSOR")


class ClinicalPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()

    # Load datasets
    def load_data(self):

        clinical_path = os.path.join(
            RAW_DATA_DIR,
            "clinical_matched.xlsx"
        )

        cbcl_path = os.path.join(
            RAW_DATA_DIR,
            "cbcl.xlsx"
        )

        clinical = pd.read_excel(clinical_path)
        cbcl = pd.read_excel(cbcl_path)

        validate_dataframe(clinical)
        validate_dataframe(cbcl)

        logger.info(f"Clinical dataset shape: {clinical.shape}")
        logger.info(f"CBCL dataset shape: {cbcl.shape}")

        return clinical, cbcl

    # Clean dataset

    def clean(self, df):

        df = df.drop_duplicates()

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(
                df[numeric_cols].mean()
            )

        return df

    def scale(self, df):

        numeric = df.select_dtypes(include="number")

        if numeric.shape[1] == 0:

            logger.warning(
                "No numeric columns found. Skipping scaling."
            )

            return df

        scaled_values = self.scaler.fit_transform(numeric)

        scaled_df = pd.DataFrame(
            scaled_values,
            columns=numeric.columns
        )

        return scaled_df

    # Main processing pipeline

    def process(self):

        clinical, cbcl = self.load_data()

        clinical = self.clean(clinical)
        cbcl = self.clean(cbcl)

        clinical_scaled = self.scale(clinical)
        cbcl_scaled = self.scale(cbcl)

        clinical_output = os.path.join(
            PROCESSED_DATA_DIR,
            "clinical_features.csv"
        )

        cbcl_output = os.path.join(
            PROCESSED_DATA_DIR,
            "cbcl_features.csv"
        )

        clinical_scaled.to_csv(
            clinical_output,
            index=False
        )

        cbcl_scaled.to_csv(
            cbcl_output,
            index=False
        )

        logger.info("Clinical preprocessing completed")
        logger.info(f"Saved: {clinical_output}")
        logger.info(f"Saved: {cbcl_output}")


# Entry point

def run():

    processor = ClinicalPreprocessor()

    processor.process()


if __name__ == "__main__":
    run()
