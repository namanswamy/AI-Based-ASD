import pandas as pd

from evaluation.evaluate_models import run
from utils.logger import get_logger

logger = get_logger("MODEL_COMPARISON")


def compare():

    results = run()

    df = pd.DataFrame(results).T

    df = df.sort_values(
        by="f1_score",
        ascending=False
    )

    logger.info("\nModel ranking:\n")

    logger.info(df)

    df.to_csv("outputs/reports/model_comparison.csv")

    return df


if __name__ == "__main__":

    compare()