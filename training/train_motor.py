import torch
import pandas as pd
import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.lstm_models import LSTMClassifier
from training.trainer_utils import train_epoch, evaluate

from utils.model_utils import save_torch_model
from utils.logger import get_logger
from utils.config import PROCESSED_DATA_DIR

logger = get_logger("TRAIN_MOTOR")


def load_data():

    import pandas as pd
    import os

    from utils.config import PROCESSED_DATA_DIR

    path = os.path.join(
        PROCESSED_DATA_DIR,
        "motor_features.csv"
    )

    df = pd.read_csv(path)

    # If label does not exist, create a dummy one
    if "label" not in df.columns:

        print("⚠️ Label column not found. Creating placeholder labels.")

        df["label"] = 0

    X = df.drop(columns=["label"]).values

    y = df["label"].values

    return X, y


def run():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    X_train = X_train[:, None, :]
    X_test = X_test[:, None, :]

    train_ds = TensorDataset(
        torch.tensor(X_train).float(),
        torch.tensor(y_train).long()
    )

    test_ds = TensorDataset(
        torch.tensor(X_test).float(),
        torch.tensor(y_test).long()
    )

    train_loader = DataLoader(train_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = LSTMClassifier(input_dim=X_train.shape[2]).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):

        loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        logger.info(f"Epoch {epoch} loss {loss}")

    metrics = evaluate(model, test_loader, device)

    logger.info(metrics)

    save_torch_model(model, "motor_lstm")


if __name__ == "__main__":

    run()