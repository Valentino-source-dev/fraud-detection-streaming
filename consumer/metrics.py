"""Prometheus metrics for the stream consumer.

Exposes an HTTP endpoint at /metrics for Prometheus to scrape.
"""

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server


# ── Counters ────────────────────────────────────────────────────
transactions_processed = Counter(
    "transactions_processed_total",
    "Total transactions consumed and scored",
)

predictions_total = Counter(
    "predictions_total",
    "Total predictions by outcome",
    ["prediction"],  # labels: fraud / legit
)

# ── Histograms ──────────────────────────────────────────────────
inference_latency = Histogram(
    "inference_latency_seconds",
    "Time spent on model inference (seconds)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

anomaly_score_distribution = Histogram(
    "anomaly_score_distribution",
    "Distribution of model anomaly scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ── Gauges ──────────────────────────────────────────────────────
fraud_rate = Gauge(
    "fraud_rate_ratio",
    "Rolling ratio of fraud predictions (updated per batch)",
)

consumer_lag = Gauge(
    "consumer_lag",
    "Estimated consumer lag in messages",
)

# ── Info ────────────────────────────────────────────────────────
model_info = Info(
    "active_model",
    "Currently loaded model metadata",
)


def start_metrics_server(port: int) -> None:
    """Start the Prometheus HTTP server on the given port."""
    start_http_server(port)
