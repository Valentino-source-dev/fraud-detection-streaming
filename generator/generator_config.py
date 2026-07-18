"""Configuration for the transaction generator."""

import os


KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:19092")
KAFKA_TOPIC: str = os.getenv("KAFKA_TOPIC", "transactions")
REPLAY_SPEED: float = float(os.getenv("REPLAY_SPEED", "10"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
DATA_PATH: str = os.getenv("DATA_PATH", "../data/creditcard.csv")
