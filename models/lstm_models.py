import torch
import torch.nn as nn

from utils.config import LSTM_HIDDEN_DIM, LSTM_LAYERS
from utils.logger import get_logger

logger = get_logger("LSTM_MODEL")


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, num_classes=2):

        super(LSTMClassifier, self).__init__()

        self.hidden_dim = LSTM_HIDDEN_DIM

        self.num_layers = LSTM_LAYERS

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc1 = nn.Linear(self.hidden_dim, 32)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(32, num_classes)

        logger.info("LSTM model initialized")

    def forward(self, x):

        batch_size = x.size(0)

        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim
        ).to(x.device)

        c0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim
        ).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.fc1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out