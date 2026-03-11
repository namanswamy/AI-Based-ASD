import numpy as np

from utils.logger import get_logger

logger = get_logger("FUSION_MODEL")


class FusionModel:

    def __init__(self, tabular_model, motor_model, eye_model):

        self.tabular_model = tabular_model

        self.motor_model = motor_model

        self.eye_model = eye_model

        logger.info("Fusion model created")

    # Tabular prediction

    def predict_tabular(self, X):

        prob = self.tabular_model.predict_proba(X)[:, 1]

        return prob

    # Motor prediction

    def predict_motor(self, model, sequence):

        model.eval()

        with torch.no_grad():

            logits = model(sequence)

            prob = torch.softmax(logits, dim=1)[:, 1]

        return prob.cpu().numpy()

    # Eye prediction

    def predict_eye(self, model, sequence):

        model.eval()

        with torch.no_grad():

            logits = model(sequence)

            prob = torch.softmax(logits, dim=1)[:, 1]

        return prob.cpu().numpy()

    # Multimodal fusion

    def predict(self, tabular, motor, eye):

        p1 = self.predict_tabular(tabular)

        p2 = self.predict_motor(self.motor_model, motor)

        p3 = self.predict_eye(self.eye_model, eye)

        final = (0.4 * p1) + (0.3 * p2) + (0.3 * p3)

        return final
