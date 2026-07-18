"""Real-time feature engineering for the fraud detection consumer.

Maintains per-card state in memory to compute rolling features on top of the
raw PCA features (V1..V28) already present in the dataset.

Because the Kaggle dataset does not include a card ID, we synthesise one
from a hash of the first few V-features to create ~1000 virtual cards.
This is realistic enough to demonstrate stateful feature engineering.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

import numpy as np


# Maximum history length per card (bounded memory)
_MAX_HISTORY = 200


class FeatureEngineer:
    """Stateful feature engineering — one instance per consumer."""

    def __init__(self) -> None:
        # Per-card state: amount history, timestamps
        self._amounts: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_HISTORY)
        )
        self._timestamps: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_HISTORY)
        )

    @staticmethod
    def _assign_card_id(event: dict[str, Any]) -> str:
        """Derive a synthetic card ID from PCA features.

        Uses a deterministic hash of V1-V5 discretised to 3 decimals so the
        same 'card' shows up across the replay.  Produces ~1 000 virtual cards.
        """
        parts = []
        for i in range(1, 6):
            v = event.get(f"V{i}", 0.0)
            parts.append(f"{v:.3f}")
        raw = "|".join(parts)
        return f"card_{abs(hash(raw)) % 10_000:05d}"

    def compute(self, event: dict[str, Any]) -> dict[str, Any]:
        """Add real-time features to *event* dict and return augmented copy.

        Added features:
          card_id            – synthetic card identifier
          amount_zscore      – z-score of Amount vs card's rolling history
          time_since_last_tx – seconds since the card's previous transaction
          tx_count_recent    – number of transactions in the last 60 s
          amount_to_mean     – Amount / rolling mean for the card (>1 = above avg)
          amount_max_ratio   – Amount / rolling max  (>1 = new max)
        """
        card_id = self._assign_card_id(event)
        amount = float(event.get("amount", 0.0))
        now = float(event.get("time", 0.0))

        amounts = self._amounts[card_id]
        timestamps = self._timestamps[card_id]

        # ── Compute features BEFORE appending current tx ────────
        features: dict[str, Any] = {}
        features["card_id"] = card_id

        # Amount z-score
        if len(amounts) >= 2:
            mean = float(np.mean(amounts))
            std = float(np.std(amounts))
            features["amount_zscore"] = (amount - mean) / std if std > 0 else 0.0
        else:
            features["amount_zscore"] = 0.0

        # Time since last transaction
        if timestamps:
            features["time_since_last_tx"] = now - timestamps[-1]
        else:
            features["time_since_last_tx"] = 0.0

        # Transaction count in last 60 s
        cutoff = now - 60.0
        features["tx_count_recent"] = sum(1 for t in timestamps if t >= cutoff)

        # Amount relative to rolling mean
        if amounts:
            rolling_mean = float(np.mean(amounts))
            features["amount_to_mean"] = amount / rolling_mean if rolling_mean > 0 else 1.0
        else:
            features["amount_to_mean"] = 1.0

        # Amount relative to rolling max
        if amounts:
            rolling_max = float(np.max(amounts))
            features["amount_max_ratio"] = amount / rolling_max if rolling_max > 0 else 1.0
        else:
            features["amount_max_ratio"] = 1.0

        # ── Update state AFTER computing features ───────────────
        amounts.append(amount)
        timestamps.append(now)

        # Merge into a copy of the original event
        augmented = {**event, **features}
        return augmented

    @property
    def active_cards(self) -> int:
        """Return the number of cards with stored state."""
        return len(self._amounts)
