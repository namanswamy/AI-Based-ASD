import os
import joblib
import torch
import json

from utils.config import SAVED_MODEL_DIR
from utils.logger import get_logger

logger = get_logger("MODEL_UTILS")


# ============================================================
# Save classical ML models
# ============================================================

def save_sklearn_model(model, name):
    """
    Save sklearn / xgboost / lightgbm models
    """

    path = os.path.join(SAVED_MODEL_DIR, f"{name}.joblib")

    joblib.dump(model, path)

    logger.info(f"Model saved -> {path}")

    return path


# ============================================================
# Load classical ML models
# ============================================================

def load_sklearn_model(name):
    """
    Load sklearn model
    """

    path = os.path.join(SAVED_MODEL_DIR, f"{name}.joblib")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    model = joblib.load(path)

    logger.info(f"Model loaded -> {path}")

    return model


# ============================================================
# Save PyTorch model
# ============================================================

def save_torch_model(model, name):
    """
    Save PyTorch model weights
    """

    path = os.path.join(SAVED_MODEL_DIR, f"{name}.pt")

    torch.save(model.state_dict(), path)

    logger.info(f"PyTorch model saved -> {path}")

    return path


# ============================================================
# Load PyTorch model
# ============================================================

def load_torch_model(model_class, name, device="cpu"):
    """
    Load PyTorch model weights
    """

    path = os.path.join(SAVED_MODEL_DIR, f"{name}.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    model = model_class()

    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    model.to(device)

    model.eval()

    logger.info(f"PyTorch model loaded -> {path}")

    return model


# ============================================================
# Save training checkpoint
# ============================================================

def save_checkpoint(model, optimizer, epoch, name):
    """
    Save full training checkpoint
    """

    path = os.path.join(SAVED_MODEL_DIR, f"{name}_checkpoint.pt")

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(checkpoint, path)

    logger.info(f"Checkpoint saved -> {path}")


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(model, optimizer, name, device="cpu"):
    """
    Resume training from checkpoint
    """

    path = os.path.join(SAVED_MODEL_DIR, f"{name}_checkpoint.pt")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    logger.info(f"Checkpoint loaded from epoch {epoch}")

    return model, optimizer, epoch


# ============================================================
# Count parameters
# ============================================================

def count_parameters(model):
    """
    Count trainable parameters in PyTorch model
    """

    return sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Save experiment metadata
# ============================================================

def save_experiment_metadata(metadata, name):
    """
    Save experiment information
    """

    path = os.path.join(SAVED_MODEL_DIR, f"{name}_metadata.json")

    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Metadata saved -> {path}")


# ============================================================
# Load experiment metadata
# ============================================================

def load_experiment_metadata(name):

    path = os.path.join(SAVED_MODEL_DIR, f"{name}_metadata.json")

    with open(path) as f:
        data = json.load(f)

    return data