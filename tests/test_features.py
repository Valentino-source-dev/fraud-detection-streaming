"""Tests for the real-time feature engineering module."""

import time
import pytest
from consumer.features import FeatureEngineer


def _make_event(**overrides) -> dict:
    """Create a minimal event dict with defaults."""
    base = {
        "original_index": 0,
        "timestamp": "2024-01-01T00:00:00Z",
        "time": 0.0,
        "amount": 100.0,
        "is_fraud": 0,
    }
    # Add V1..V28
    for i in range(1, 29):
        base[f"V{i}"] = float(i) * 0.1
    base.update(overrides)
    return base


class TestFeatureEngineer:
    def test_first_event_has_zero_defaults(self):
        """First event for a card should have zeroed rolling features."""
        fe = FeatureEngineer()
        result = fe.compute(_make_event())

        assert result["amount_zscore"] == 0.0
        assert result["time_since_last_tx"] == 0.0
        assert result["tx_count_recent"] == 0
        assert result["amount_to_mean"] == 1.0
        assert result["amount_max_ratio"] == 1.0

    def test_card_id_is_deterministic(self):
        """Same V1-V5 values should produce the same card ID."""
        fe = FeatureEngineer()
        e1 = fe.compute(_make_event())
        e2 = fe.compute(_make_event())
        assert e1["card_id"] == e2["card_id"]

    def test_different_features_different_card(self):
        """Different V1 values should produce different card IDs."""
        fe = FeatureEngineer()
        e1 = fe.compute(_make_event(V1=1.0))
        e2 = fe.compute(_make_event(V1=999.0))
        assert e1["card_id"] != e2["card_id"]

    def test_zscore_after_multiple_events(self):
        """After several events, z-score should be computed."""
        fe = FeatureEngineer()
        # Send events with varying amounts so std > 0, and explicit times
        for i in range(5):
            fe.compute(_make_event(amount=100.0 + (i % 2) * 10, time=float(i)))

        # Now send a high-amount event
        result = fe.compute(_make_event(amount=500.0, time=5.0))
        assert result["amount_zscore"] > 0, "High amount should have positive z-score"

    def test_time_since_last_tx(self):
        """time_since_last_tx should reflect simulated elapsed time."""
        fe = FeatureEngineer()
        fe.compute(_make_event(time=1.0))
        result = fe.compute(_make_event(time=1.05))
        assert result["time_since_last_tx"] == pytest.approx(0.05)

    def test_tx_count_recent(self):
        """tx_count_recent should count events within the last 60s of simulated time."""
        fe = FeatureEngineer()
        for i in range(10):
            fe.compute(_make_event(time=float(i) * 2.0)) # spans 18s
        result = fe.compute(_make_event(time=20.0))
        # All 10 prior events are within [20 - 60, 20)
        assert result["tx_count_recent"] == 10

    def test_amount_max_ratio(self):
        """A new maximum amount should produce amount_max_ratio > 1."""
        fe = FeatureEngineer()
        fe.compute(_make_event(amount=50.0, time=1.0))
        fe.compute(_make_event(amount=50.0, time=2.0))
        result = fe.compute(_make_event(amount=100.0, time=3.0))
        assert result["amount_max_ratio"] == 2.0

    def test_active_cards_count(self):
        """active_cards should track unique cards."""
        fe = FeatureEngineer()
        fe.compute(_make_event(V1=1.0))
        fe.compute(_make_event(V1=2.0))
        fe.compute(_make_event(V1=3.0))
        assert fe.active_cards >= 2  # at least 2 different cards

    def test_original_event_fields_preserved(self):
        """Enriched event should contain all original fields."""
        fe = FeatureEngineer()
        result = fe.compute(_make_event(amount=42.0, is_fraud=1))
        assert result["amount"] == 42.0
        assert result["is_fraud"] == 1
        assert "V1" in result
        assert "V28" in result

    def test_training_serving_feature_equivalence(self):
        """Verifica che compute_batch_features e FeatureEngineer.compute
        producano esattamente gli stessi valori per evitare skew.
        """
        import pytest
        import pandas as pd
        from training.features import compute_batch_features
        
        # 1. Genera una sequenza coerente di eventi
        events = [
            _make_event(time=1.0, amount=10.0, V1=0.1, V2=0.2, V3=0.3, V4=0.4, V5=0.5),
            _make_event(time=5.0, amount=20.0, V1=0.1, V2=0.2, V3=0.3, V4=0.4, V5=0.5),
            _make_event(time=12.0, amount=15.0, V1=0.1, V2=0.2, V3=0.3, V4=0.4, V5=0.5)
        ]
        
        # 2. Calcola in streaming
        fe = FeatureEngineer()
        streaming_results = [fe.compute(e) for e in events]
        
        df = pd.DataFrame(events)
        # Adatta i nomi colonne per uniformare Amount/amount e Time/time
        df = df.rename(columns={"amount": "Amount", "time": "Time"})
        batch_results = compute_batch_features(df)
        
        # 4. Confronta i valori
        for i, st_res in enumerate(streaming_results):
            row = batch_results.iloc[i]
            assert st_res["amount_zscore"] == pytest.approx(row["amount_zscore"])
            assert st_res["time_since_last_tx"] == pytest.approx(row["time_since_last_tx"])
            assert st_res["tx_count_recent"] == int(row["tx_count_recent"])
            assert st_res["amount_to_mean"] == pytest.approx(row["amount_to_mean"])
            assert st_res["amount_max_ratio"] == pytest.approx(row["amount_max_ratio"])

