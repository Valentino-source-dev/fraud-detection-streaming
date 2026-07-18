"""Stream Consumer — Kafka consumer with real-time feature engineering & inference.

Consumes transactions from Kafka, enriches them with stateful features, runs
the fraud detection model, and logs results to PostgreSQL + Prometheus.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time

import numpy as np
from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

import consumer_config as config
import metrics
from db import PredictionWriter
from features import FeatureEngineer
from model_loader import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("consumer")

_shutdown = False


def _handle_signal(signum: int, _frame) -> None:
    global _shutdown
    log.info("Received signal %d — shutting down gracefully.", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Feature vector construction ─────────────────────────────────

# The V1..V28 features from the dataset + the real-time features we compute.
# The model was trained on these exact columns (see training/train.py).
_V_FEATURES = [f"V{i}" for i in range(1, 29)]
_EXTRA_FEATURES = [
    "amount",
    "amount_zscore",
    "time_since_last_tx",
    "tx_count_recent",
    "amount_to_mean",
    "amount_max_ratio",
]
FEATURE_COLUMNS = _V_FEATURES + _EXTRA_FEATURES


def _build_feature_vector(event: dict) -> np.ndarray:
    """Extract a 1D numpy array of features from an enriched event dict."""
    values = [event.get(col, 0.0) for col in FEATURE_COLUMNS]
    return np.array(values, dtype=np.float64).reshape(1, -1)


# ── Main loop ───────────────────────────────────────────────────

def _wait_for_broker(broker: str, max_retries: int = 30, delay: float = 2.0) -> Consumer:
    """Block until the Kafka broker is reachable, then return a Consumer."""
    consumer_conf = {
        "bootstrap.servers": broker,
        "group.id": config.CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "auto.commit.interval.ms": 1000,
    }
    for attempt in range(1, max_retries + 1):
        try:
            consumer = Consumer(consumer_conf)
            consumer.list_topics(timeout=5)
            log.info("Connected to broker %s on attempt %d.", broker, attempt)
            return consumer
        except KafkaException as exc:
            log.warning("Broker not ready (attempt %d/%d): %s", attempt, max_retries, exc)
            time.sleep(delay)

    log.critical("Could not connect to broker after %d attempts.", max_retries)
    sys.exit(1)


def run() -> None:
    """Main consumer loop."""
    # Start Prometheus metrics server
    metrics.start_metrics_server(config.METRICS_PORT)
    log.info("Prometheus metrics server on port %d.", config.METRICS_PORT)

    # Initialise components
    consumer = _wait_for_broker(config.KAFKA_BROKER)
    consumer.subscribe([config.KAFKA_TOPIC])
    log.info("Subscribed to topic '%s'.", config.KAFKA_TOPIC)

    feature_eng = FeatureEngineer()
    model_mgr = ModelManager()
    db_writer = PredictionWriter()

    processed = 0
    frauds_detected = 0
    recent_predictions: list[bool] = []  # rolling window for fraud rate

    try:
        while not _shutdown:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                # No message — flush DB buffer if stale
                db_writer.flush()
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                log.error("Consumer error: %s", msg.error())
                continue

            # ── Parse message ──────────────────────────────
            try:
                event = json.loads(msg.value().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning("Skipping malformed message: %s", exc)
                continue

            # ── Feature engineering ────────────────────────
            enriched = feature_eng.compute(event)

            # ── Model inference ────────────────────────────
            scorer = model_mgr.ensure_loaded()
            feature_vec = _build_feature_vector(enriched)

            t0 = time.perf_counter()
            try:
                proba = scorer.predict_proba(feature_vec)
                # proba shape: (1, 2) — [P(legit), P(fraud)]
                fraud_score = float(proba[0, 1]) if proba.shape[1] == 2 else float(proba[0, 0])
            except Exception as exc:
                log.error("Inference failed: %s", exc)
                fraud_score = 0.0
            latency_ms = (time.perf_counter() - t0) * 1000

            is_fraud_pred = fraud_score >= config.SCORE_THRESHOLD

            # ── Update Prometheus metrics ──────────────────
            metrics.transactions_processed.inc()
            metrics.inference_latency.observe(latency_ms / 1000)
            metrics.anomaly_score_distribution.observe(fraud_score)
            label = "fraud" if is_fraud_pred else "legit"
            metrics.predictions_total.labels(prediction=label).inc()

            # Rolling fraud rate (last 1000 predictions)
            recent_predictions.append(is_fraud_pred)
            if len(recent_predictions) > 1000:
                recent_predictions.pop(0)
            current_fraud_rate = sum(recent_predictions) / len(recent_predictions)
            metrics.fraud_rate.set(current_fraud_rate)

            # Compute and update consumer lag
            try:
                tp = TopicPartition(msg.topic(), msg.partition())
                _, high_offset = consumer.get_watermark_offsets(tp)
                lag = max(0, high_offset - msg.offset() - 1)
                metrics.consumer_lag.set(lag)
            except Exception as exc:
                log.debug("Could not compute consumer lag: %s", exc)

            # Model info
            metrics.model_info.info({"version": getattr(scorer, "version", "unknown")})

            # ── Log to PostgreSQL ──────────────────────────
            db_writer.add({
                "original_index": event.get("original_index"),
                "amount": event.get("amount", 0.0),
                "score": fraud_score,
                "is_fraud_pred": is_fraud_pred,
                "is_fraud_actual": bool(event.get("is_fraud", 0)),
                "model_version": getattr(scorer, "version", "unknown"),
                "latency_ms": latency_ms,
                "features": {k: enriched.get(k) for k in _EXTRA_FEATURES},
            })

            processed += 1
            if is_fraud_pred:
                frauds_detected += 1

            # Periodic log
            if processed % 500 == 0:
                log.info(
                    "Processed %d | frauds detected: %d | fraud rate: %.3f%% | "
                    "latency: %.2fms | active cards: %d",
                    processed,
                    frauds_detected,
                    current_fraud_rate * 100,
                    latency_ms,
                    feature_eng.active_cards,
                )

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        log.info("Shutting down consumer...")
        consumer.close()
        db_writer.close()
        log.info(
            "Consumer stopped. Total processed: %d, frauds detected: %d.",
            processed,
            frauds_detected,
        )


if __name__ == "__main__":
    run()
