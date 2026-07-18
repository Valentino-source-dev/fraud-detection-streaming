"""PostgreSQL writer — batched inserts for prediction results.

Accumulates predictions in a buffer and flushes periodically or when the
buffer reaches DB_BATCH_SIZE.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import psycopg2
import psycopg2.extras

import consumer_config as config

log = logging.getLogger("consumer.db")


INSERT_SQL = """
    INSERT INTO predictions
        (original_index, amount, score, is_fraud_pred, is_fraud_actual,
         model_version, latency_ms, features_json)
    VALUES %s
"""


class PredictionWriter:
    """Batched PostgreSQL writer for prediction results."""

    def __init__(self) -> None:
        self._buffer: list[tuple] = []
        self._last_flush: float = time.monotonic()
        self._conn: psycopg2.extensions.connection | None = None

    def _connect(self) -> psycopg2.extensions.connection:
        """Establish or return an existing PostgreSQL connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=config.POSTGRES_HOST,
                port=config.POSTGRES_PORT,
                dbname=config.POSTGRES_DB,
                user=config.POSTGRES_USER,
                password=config.POSTGRES_PASSWORD,
            )
            self._conn.autocommit = True
            log.info("Connected to PostgreSQL at %s:%s.", config.POSTGRES_HOST, config.POSTGRES_PORT)
        return self._conn

    def add(self, prediction: dict[str, Any]) -> None:
        """Buffer a prediction dict for later batch insert.

        Expected keys: original_index, amount, score, is_fraud_pred,
                       is_fraud_actual, model_version, latency_ms, features
        """
        row = (
            prediction.get("original_index"),
            prediction.get("amount"),
            prediction["score"],
            prediction["is_fraud_pred"],
            prediction.get("is_fraud_actual"),
            prediction.get("model_version", "unknown"),
            prediction.get("latency_ms"),
            json.dumps(prediction.get("features", {})),
        )
        self._buffer.append(row)

        # Flush if buffer is full or enough time has passed
        now = time.monotonic()
        if (
            len(self._buffer) >= config.DB_BATCH_SIZE
            or now - self._last_flush >= config.DB_FLUSH_INTERVAL_SEC
        ):
            self.flush()

    def flush(self) -> None:
        """Write all buffered predictions to PostgreSQL."""
        if not self._buffer:
            return

        try:
            conn = self._connect()
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, INSERT_SQL, self._buffer)
            log.debug("Flushed %d predictions to PostgreSQL.", len(self._buffer))
        except psycopg2.Error as exc:
            log.error("Failed to flush predictions: %s", exc)
            # Reconnect on next attempt
            self._conn = None
        finally:
            self._buffer.clear()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        """Flush remaining data and close the connection."""
        self.flush()
        if self._conn and not self._conn.closed:
            self._conn.close()
            log.info("PostgreSQL connection closed.")
