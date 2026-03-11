import os

# ======================================
# Project Root
# ======================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# ======================================
# Data directories
# ======================================

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# ======================================
# Models
# ======================================

MODEL_DIR = os.path.join(BASE_DIR, "models")

SAVED_MODEL_DIR = os.path.join(MODEL_DIR, "saved")

# ======================================
# Outputs
# ======================================

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

LOG_DIR = os.path.join(BASE_DIR, "logs")

EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")

# ======================================
# Create folders automatically
# ======================================

for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODEL_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    EXPERIMENT_DIR
]:
    os.makedirs(path, exist_ok=True)

# ======================================
# Model parameters
# ======================================

RANDOM_STATE = 42

TABULAR_TEST_SIZE = 0.2

SEQUENCE_LENGTH = 10

LSTM_HIDDEN_DIM = 64

LSTM_LAYERS = 2

DEVICE = "cpu"