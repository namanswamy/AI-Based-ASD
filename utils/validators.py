import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("VALIDATORS")


# =========================================================
# Dataset validation
# =========================================================

def validate_dataframe(df):
    """
    Basic dataframe validation
    """

    if not isinstance(df, pd.DataFrame):

        raise ValueError("Input must be a pandas DataFrame")

    if df.empty:

        raise ValueError("DataFrame is empty")

    logger.info(f"Dataset shape: {df.shape}")


# =========================================================
# Missing values
# =========================================================

def check_missing_values(df):

    missing = df.isnull().sum()

    total_missing = missing.sum()

    logger.info(f"Missing values count: {total_missing}")

    return missing


# =========================================================
# Numeric validation
# =========================================================

def ensure_numeric(df):

    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(df[col]):

            raise ValueError(
                f"Column {col} must be numeric"
            )

    logger.info("All columns numeric")


# =========================================================
# Label validation
# =========================================================

def validate_labels(y):

    unique = np.unique(y)

    if len(unique) < 2:

        raise ValueError("Labels must have at least two classes")

    logger.info(f"Label classes: {unique}")


# =========================================================
# Sequence validation (for LSTM)
# =========================================================

def validate_sequence_data(X):

    if len(X.shape) != 3:

        raise ValueError(
            "Sequence data must be 3D: (samples, time_steps, features)"
        )

    logger.info(
        f"Sequence dataset shape: {X.shape}"
    )


# =========================================================
# Feature sanity check
# =========================================================

def validate_feature_range(df):

    if np.isinf(df.values).any():

        raise ValueError("Infinite values detected")

    if np.isnan(df.values).any():

        raise ValueError("NaN values detected")

    logger.info("Feature values validated")


# =========================================================
# Train/Test split validation
# =========================================================

def validate_split(X_train, X_test, y_train, y_test):

    if len(X_train) == 0 or len(X_test) == 0:

        raise ValueError("Invalid train/test split")

    if len(y_train) != len(X_train):

        raise ValueError("Train data mismatch")

    if len(y_test) != len(X_test):

        raise ValueError("Test data mismatch")

    logger.info("Train/Test split validated")