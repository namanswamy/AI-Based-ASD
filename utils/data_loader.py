import os
import pandas as pd

from utils.config import RAW_DATA_DIR


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