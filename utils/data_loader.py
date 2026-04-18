import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE


# =========================================================
# Columns to DROP — leaking the target or identifiers
# =========================================================

LEAKING_COLUMNS = [
    "Autism_Diagnosis",   # directly encodes the target
    "ASD_Severity",       # derived from diagnosis
    "Therapy_Progress",   # post-diagnosis, leaks target
    "result",             # AQ-10 total — perfectly separates classes (no overlap)
    "used_app_before",    # metadata artifact, corr=-0.90 with target
]

ID_COLUMNS = [
    "id",
    "ID",
    "participant_id",
    "Unnamed: 0",
    "Unnamed_0",
    "Class/ASD.1",        # duplicate target from concat
    "Class_ASD_1",        # sanitized version of above
    "label",              # secondary label column (constant)
]

DROP_COLUMNS = LEAKING_COLUMNS + ID_COLUMNS

TARGET = "Class/ASD"


# =========================================================
# Load and clean the multimodal dataset
# =========================================================

def load_clean_dataset():
    """Load the multimodal dataset with leaking/ID columns removed.

    The merged dataset contains rows from multiple source datasets
    concatenated horizontally. Class 2 comes from a different source
    than classes 0/1, creating trivially separable NaN patterns.
    We filter to the binary ASD screening task (classes 0 vs 1) and
    drop columns that are >80% NaN in that subset.
    """
    path = os.path.join(PROCESSED_DATA_DIR, "multimodal_dataset.csv")
    df = pd.read_csv(path)

    # Keep only binary ASD classes (0=No ASD, 1=ASD) from the
    # primary tabular dataset — class 2 is from a different source
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)

    y = df[TARGET]

    X = df.drop(columns=[TARGET], errors="ignore")
    X = X.drop(columns=[c for c in DROP_COLUMNS if c in X.columns], errors="ignore")

    # Drop columns that are mostly NaN (>80%) in the binary subset
    null_pct = X.isnull().mean()
    sparse_cols = null_pct[null_pct > 0.8].index.tolist()
    if sparse_cols:
        X = X.drop(columns=sparse_cols)

    # Sanitize column names for LightGBM/XGBoost compatibility
    X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    return X, y


def load_split_dataset(test_size=0.2):
    """Load dataset, remove leaking features, impute NaNs, and split."""
    X, y = load_clean_dataset()

    # Drop columns that are entirely NaN before imputing
    all_nan_cols = X.columns[X.isnull().all()]
    if len(all_nan_cols) > 0:
        X = X.drop(columns=all_nan_cols)

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# Raw data loaders (unchanged)
# =========================================================

def load_autism_adult():
    path = os.path.join(RAW_DATA_DIR, "autism_adult.csv")
    return pd.read_csv(path)


def load_autism_child():
    path = os.path.join(RAW_DATA_DIR, "autism_child.csv")
    return pd.read_csv(path)


def load_autism_adolescent():
    path = os.path.join(RAW_DATA_DIR, "autism_adolescent.csv")
    return pd.read_csv(path)


def load_realistic():
    path = os.path.join(RAW_DATA_DIR, "autism_realistic.csv")
    return pd.read_csv(path)


def load_eye_tracking():
    path = os.path.join(RAW_DATA_DIR, "eye_tracking.csv")
    return pd.read_csv(path)


def load_motor_pattern():
    path = os.path.join(RAW_DATA_DIR, "motor_pattern.csv")
    return pd.read_csv(path)


def load_clinical():
    path = os.path.join(RAW_DATA_DIR, "clinical_matched.xlsx")
    return pd.read_excel(path)


def load_cbcl():
    path = os.path.join(RAW_DATA_DIR, "cbcl.xlsx")
    return pd.read_excel(path)
