"""Tests for the model loader module."""

import numpy as np
from consumer.model_loader import DummyScorer, ModelManager


class TestDummyScorer:
    def test_returns_zero_fraud_probability(self):
        scorer = DummyScorer()
        X = np.array([[1.0, 2.0, 3.0]])
        proba = scorer.predict_proba(X)
        assert proba.shape == (1, 2)
        assert proba[0, 0] == 1.0  # P(legit) = 1
        assert proba[0, 1] == 0.0  # P(fraud) = 0

    def test_batch_prediction(self):
        scorer = DummyScorer()
        X = np.random.randn(10, 5)
        proba = scorer.predict_proba(X)
        assert proba.shape == (10, 2)
        assert np.all(proba[:, 0] == 1.0)
        assert np.all(proba[:, 1] == 0.0)

    def test_version(self):
        scorer = DummyScorer()
        assert scorer.version == "dummy-v0"


class TestModelManager:
    def test_starts_with_dummy(self):
        mgr = ModelManager()
        assert isinstance(mgr.scorer, DummyScorer)

    def test_ensure_loaded_returns_scorer(self):
        mgr = ModelManager()
        # Without MLflow running, should return dummy
        scorer = mgr.ensure_loaded()
        assert hasattr(scorer, "predict_proba")
        assert hasattr(scorer, "version")
