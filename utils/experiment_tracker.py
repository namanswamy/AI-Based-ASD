import mlflow
import os

TRACKING_DIR = "experiments"

os.makedirs(TRACKING_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:{TRACKING_DIR}")


def start_experiment(name):
    mlflow.set_experiment(name)
    mlflow.start_run()


def log_metric(name, value):
    mlflow.log_metric(name, value)


def log_param(name, value):
    mlflow.log_param(name, value)


def end_experiment():
    mlflow.end_run()