from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils.config import RANDOM_STATE
from utils.logger import get_logger

logger = get_logger("CLASSICAL_MODELS")

# Random Forest

def build_random_forest():

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    logger.info("Random Forest initialized")

    return model

# Support Vector Machine

def build_svm():

    model = SVC(
        kernel="rbf",
        probability=True,
        C=2.0,
        gamma="scale"
    )

    logger.info("SVM initialized")

    return model

# XGBoost

def build_xgboost():

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE
    )

    logger.info("XGBoost initialized")

    return model

# LightGBM

def build_lightgbm():

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE
    )

    logger.info("LightGBM initialized")

    return model
