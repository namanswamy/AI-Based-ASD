import logging
import os
from datetime import datetime

from utils.config import LOG_DIR


def get_logger(name):

    log_file = os.path.join(
        LOG_DIR,
        f"{datetime.now().strftime('%Y%m%d')}.log"
    )

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger