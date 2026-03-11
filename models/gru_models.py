import torch
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger("GRU_MODEL")


class GRUClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):

        super(GRUClassifier, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 2)

        self.dropout = nn.Dropout(0.3)

        logger.info("GRU model initialized")

    def forward(self, x):

        out, _ = self.gru(x)

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.fc(out)

        return out