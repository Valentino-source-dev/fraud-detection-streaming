"""Configuration for the stream consumer."""

import os


KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:19092")
KAFKA_TOPIC: str = os.getenv("KAFKA_TOPIC", "transactions")
CONSUMER_GROUP: str = os.getenv("CONSUMER_GROUP", "fraud-detector")

SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.5"))
METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8000"))
MODEL_CHECK_INTERVAL_SEC: int = int(os.getenv("MODEL_CHECK_INTERVAL_SEC", "300"))

POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "fraud_detection")
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "fraud")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "fraud_secret_change_me")

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "fraud-detector")
MLFLOW_MODEL_STAGE: str = os.getenv("MLFLOW_MODEL_STAGE", "Production")

# Batch insert parameters
DB_BATCH_SIZE: int = int(os.getenv("DB_BATCH_SIZE", "50"))
DB_FLUSH_INTERVAL_SEC: float = float(os.getenv("DB_FLUSH_INTERVAL_SEC", "5.0"))
