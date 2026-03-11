import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger
from utils.validators import validate_dataframe

logger = get_logger("TABULAR_PREPROCESSOR")


class TabularPreprocessor:

    def __init__(self):

        self.scaler = StandardScaler()

        self.encoders = {}

    # ---------------------------------------------
    # Load datasets
    # ---------------------------------------------

    def load_datasets(self):

        datasets = [

            "autism_adult.csv",
            "autism_child.csv",
            "autism_adolescent.csv",
            "autism_realistic.csv"
        ]

        dataframes = []

        for file in datasets:

            path = os.path.join(RAW_DATA_DIR, file)

            df = pd.read_csv(path)

            validate_dataframe(df)

            logger.info(f"Loaded {file} {df.shape}")

            dataframes.append(df)

        return pd.concat(dataframes, ignore_index=True)

    # ---------------------------------------------
    # Encode categorical features
    # ---------------------------------------------

    def encode_categorical(self, df):

        for col in df.columns:

            if df[col].dtype == "object":

                le = LabelEncoder()

                df[col] = le.fit_transform(df[col].astype(str))

                self.encoders[col] = le

        return df

    # ---------------------------------------------
    # Scale numeric features
    # ---------------------------------------------

    def scale_features(self, df, label):

        X = df.drop(columns=[label])

        y = df[label]

        X_scaled = self.scaler.fit_transform(X)

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=X.columns
        )

        X_scaled[label] = y.values

        return X_scaled

    # ---------------------------------------------
    # Run pipeline
    # ---------------------------------------------

    def process(self):

        df = self.load_datasets()

        label = "Class/ASD"

        df = self.encode_categorical(df)

        df = self.scale_features(df, label)

        output_path = os.path.join(
            PROCESSED_DATA_DIR,
            "tabular_features.csv"
        )

        df.to_csv(output_path, index=False)

        logger.info(f"Saved {output_path}")


def run():

    processor = TabularPreprocessor()

    processor.process()


if __name__ == "__main__":

    run()