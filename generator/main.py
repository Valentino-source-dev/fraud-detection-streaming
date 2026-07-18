"""Transaction Generator — replays creditcard.csv as a real-time Kafka stream.

Reads the Kaggle Credit Card Fraud Detection CSV and publishes each row as a
JSON message to a Kafka topic, respecting the original inter-transaction
delays (scaled by REPLAY_SPEED).
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from confluent_kafka import Producer, KafkaError

import generator_config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("generator")

_shutdown = False


def _handle_signal(signum: int, _frame) -> None:
    global _shutdown
    log.info("Received signal %d — shutting down gracefully.", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _delivery_report(err, msg) -> None:
    """Kafka delivery callback — logs failures."""
    if err is not None:
        log.error("Delivery failed for offset %s: %s", msg.offset(), err)


def _wait_for_broker(broker: str, max_retries: int = 30, delay: float = 2.0) -> Producer:
    """Block until the Kafka broker is reachable, then return a Producer."""
    for attempt in range(1, max_retries + 1):
        try:
            producer = Producer({
                "bootstrap.servers": broker,
                "linger.ms": 50,
                "batch.num.messages": config.BATCH_SIZE,
                "queue.buffering.max.messages": 100_000,
            })
            # Trigger metadata fetch to verify connectivity
            metadata = producer.list_topics(timeout=5)
            log.info(
                "Connected to broker %s (cluster id: %s) on attempt %d.",
                broker,
                metadata.cluster_id,
                attempt,
            )
            return producer
        except KafkaError as exc:
            log.warning("Broker not ready (attempt %d/%d): %s", attempt, max_retries, exc)
            time.sleep(delay)

    log.critical("Could not connect to broker %s after %d attempts.", broker, max_retries)
    sys.exit(1)


def _load_dataset(path: str) -> pd.DataFrame:
    """Load the Credit Card Fraud CSV and validate its structure."""
    log.info("Loading dataset from %s ...", path)
    df = pd.read_csv(path)
    required_cols = {"Time", "Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        log.critical("Dataset is missing columns: %s", missing)
        sys.exit(1)
    log.info(
        "Loaded %d transactions (%d fraudulent, %.3f%% fraud rate).",
        len(df),
        df["Class"].sum(),
        df["Class"].mean() * 100,
    )
    return df


def run() -> None:
    """Main generator loop."""
    producer = _wait_for_broker(config.KAFKA_BROKER)
    df = _load_dataset(config.DATA_PATH)

    # Sort by Time to respect original ordering
    df = df.sort_values("Time").reset_index(drop=True)

    topic = config.KAFKA_TOPIC
    speed = config.REPLAY_SPEED
    published = 0
    frauds_published = 0
    start_wall = time.monotonic()

    log.info(
        "Starting replay on topic '%s' at %.1fx speed (%d rows).",
        topic,
        speed,
        len(df),
    )

    prev_time = df.iloc[0]["Time"]

    for idx, row in df.iterrows():
        if _shutdown:
            log.info("Shutdown requested after %d messages.", published)
            break

        # Simulate inter-transaction delay (scaled)
        dt = (row["Time"] - prev_time) / speed
        if dt > 0:
            # Cap the maximum sleep to avoid long pauses
            time.sleep(min(dt, 2.0))
        prev_time = row["Time"]

        # Build message
        message = {
            "original_index": int(idx),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time": float(row["Time"]),
            "amount": float(row["Amount"]),
            "is_fraud": int(row["Class"]),
        }
        # Include V1..V28 features
        for i in range(1, 29):
            col = f"V{i}"
            if col in row.index:
                message[col] = float(row[col])

        # Publish
        producer.produce(
            topic,
            key=str(idx),
            value=json.dumps(message),
            callback=_delivery_report,
        )
        published += 1
        if row["Class"] == 1:
            frauds_published += 1

        # Periodic flush & log
        if published % 1000 == 0:
            producer.flush()
            elapsed = time.monotonic() - start_wall
            rate = published / elapsed if elapsed > 0 else 0
            log.info(
                "Published %d messages (%.1f msg/s, %d frauds).",
                published,
                rate,
                frauds_published,
            )

    # Final flush
    remaining = producer.flush(timeout=30)
    elapsed = time.monotonic() - start_wall
    log.info(
        "Done. Published %d messages in %.1fs (%.1f msg/s). Frauds: %d. Unflushed: %d.",
        published,
        elapsed,
        published / elapsed if elapsed > 0 else 0,
        frauds_published,
        remaining,
    )


if __name__ == "__main__":
    run()
