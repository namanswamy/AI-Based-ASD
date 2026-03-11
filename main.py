from utils.logger import get_logger
from utils.helpers import seed_everything

logger = get_logger("ASD_MAIN")


def main():

    seed_everything()

    logger.info("ASD AI Research System Starting")


if __name__ == "__main__":

    main()