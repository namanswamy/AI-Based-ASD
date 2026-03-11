import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

from utils.config import PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger("TRAIN_EYE")


def load_data():

    path = os.path.join(
        PROCESSED_DATA_DIR,
        "eye_features.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "eye_features.csv not found. Run preprocessing first."
        )

    df = pd.read_csv(path)

    logger.info(f"Loaded dataset {df.shape}")

    # Ensure label exists
    if "Class/ASD" not in df.columns:
        raise ValueError("Dataset missing label column Class/ASD")

    X = df.drop(columns=["Class/ASD"])
    y = df["Class/ASD"]

    return X, y


def run():

    X, y = load_data()

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    logger.info("Training eye-feature model")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    report = classification_report(y_test, preds)

    logger.info("\n" + report)

    save_dir = os.path.join(
        "models",
        "saved"
    )

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        "eye_model.joblib"
    )

    joblib.dump(model, save_path)

    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    run()