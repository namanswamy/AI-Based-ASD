import torch
import numpy as np

from utils.metrics import classification_metrics
from utils.logger import get_logger

logger = get_logger("TRAINER_UTILS")


def train_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0

    for xb, yb in loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):

    model.eval()

    preds = []
    probs = []
    targets = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device)

            logits = model(xb)

            p = torch.softmax(logits, dim=1)[:, 1]

            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            probs.extend(p.cpu().numpy())
            targets.extend(yb.numpy())

    metrics = classification_metrics(
        np.array(targets),
        np.array(preds),
        np.array(probs)
    )

    return metrics