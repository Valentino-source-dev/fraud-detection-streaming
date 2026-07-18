"""Configuration for offline training scripts."""

import os


DATA_PATH: str = os.getenv("DATA_PATH", "../data/creditcard.csv")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection-v3")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "fraud-detector")
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
OPTUNA_TRIALS: int = int(os.getenv("OPTUNA_TRIALS", "50"))
