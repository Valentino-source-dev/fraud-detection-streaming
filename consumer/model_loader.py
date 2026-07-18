"""Model loader — fetches the production model from MLflow.

Falls back to a dummy scorer if MLflow is unavailable or the model is not yet
registered (e.g. first run before training).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import numpy as np

import consumer_config as config

log = logging.getLogger("consumer.model_loader")


# ── Public interface ────────────────────────────────────────────

class Scorer(Protocol):
    """Anything that can score a feature vector."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class DummyScorer:
    """Fallback scorer that flags nothing — used before a model is trained."""

    version: str = "dummy-v0"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return zero fraud probability for all inputs."""
        n = X.shape[0] if X.ndim > 1 else 1
        # Return shape (n, 2): [P(legit), P(fraud)]
        probs = np.zeros((n, 2))
        probs[:, 0] = 1.0
        return probs


class MLflowScorer:
    """Wraps an MLflow model that exposes predict_proba."""

    def __init__(self, model: Any, version: str) -> None:
        self._model = model
        self.version = version

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


def load_model() -> DummyScorer | MLflowScorer:
    """Try to load the production model from MLflow; fall back to dummy.

    Returns either an MLflowScorer or a DummyScorer.
    """
    try:
        import mlflow
        import mlflow.xgboost

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        model_uri = f"models:/{config.MLFLOW_MODEL_NAME}@production"
        log.info("Loading model from MLflow: %s", model_uri)

        # Use native xgboost loader to ensure predict_proba is directly available
        model = mlflow.xgboost.load_model(model_uri)

        # Attempt to get version info
        client = mlflow.tracking.MlflowClient()
        try:
            mv = client.get_model_version_by_alias(config.MLFLOW_MODEL_NAME, "production")
            version_str = mv.version
        except Exception:
            version_str = "unknown"

        scorer = MLflowScorer(model, version=version_str)
        log.info("Model loaded successfully (version %s).", version_str)
        return scorer

    except Exception as exc:
        log.warning(
            "Could not load model from MLflow (%s). Using dummy scorer.", exc
        )
        return DummyScorer()


class ModelManager:
    """Manages model lifecycle: loading, periodic refresh checks."""

    def __init__(self) -> None:
        self.scorer: DummyScorer | MLflowScorer = DummyScorer()
        self._last_check: float = 0.0
        self._check_interval: float = float(config.MODEL_CHECK_INTERVAL_SEC)

    def ensure_loaded(self) -> DummyScorer | MLflowScorer:
        """Return the current scorer, refreshing if the check interval elapsed."""
        now = time.monotonic()
        if now - self._last_check >= self._check_interval:
            self.scorer = load_model()
            self._last_check = now
        return self.scorer
