import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from utils.logger import get_logger

logger = get_logger("METRICS")


# =========================================================
# Classification Metrics
# =========================================================

def classification_metrics(y_true, y_pred, y_prob=None):

    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )

    results = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)

    # detect binary vs multiclass
    unique_labels = np.unique(y_true)

    if len(unique_labels) == 2:

        avg = "binary"

    else:

        avg = "weighted"

    results["precision"] = precision_score(
        y_true,
        y_pred,
        average=avg,
        zero_division=0
    )

    results["recall"] = recall_score(
        y_true,
        y_pred,
        average=avg,
        zero_division=0
    )

    results["f1_score"] = f1_score(
        y_true,
        y_pred,
        average=avg,
        zero_division=0
    )

    # ROC only for binary
    if y_prob is not None and len(unique_labels) == 2:

        results["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )

    return results


# =========================================================
# Regression Metrics
# =========================================================

def regression_metrics(y_true, y_pred):
    """
    Metrics for regression tasks
    """

    results = {}

    results["mse"] = mean_squared_error(y_true, y_pred)

    results["rmse"] = np.sqrt(results["mse"])

    results["mae"] = mean_absolute_error(y_true, y_pred)

    results["r2"] = r2_score(y_true, y_pred)

    logger.info(f"Regression Metrics: {results}")

    return results


# =========================================================
# Confusion Matrix
# =========================================================

def compute_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"Confusion Matrix:\n{cm}")

    return cm


# =========================================================
# Full Classification Report
# =========================================================

def compute_classification_report(y_true, y_pred):

    report = classification_report(y_true, y_pred)

    logger.info(f"Classification Report:\n{report}")

    return report


# =========================================================
# Model Comparison Utility
# =========================================================

def compare_models(results_dict):
    """
    Compare multiple models based on F1 score
    """

    sorted_models = sorted(
        results_dict.items(),
        key=lambda x: x[1]["f1_score"],
        reverse=True
    )

    logger.info("Model Comparison Ranking:")

    for name, metrics in sorted_models:

        logger.info(f"{name} -> F1={metrics['f1_score']}")

    return sorted_models