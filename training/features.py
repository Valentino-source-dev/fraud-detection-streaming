"""Batch feature engineering for training.

Mirrors the real-time FeatureEngineer in consumer/features.py but operates
on a full DataFrame at once.  Both implementations must produce the same
feature set to avoid training/serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Feature columns produced by this module (must match consumer/features.py)
EXTRA_FEATURES = [
    "amount_zscore",
    "time_since_last_tx",
    "tx_count_recent",
    "amount_to_mean",
    "amount_max_ratio",
]

V_FEATURES = [f"V{i}" for i in range(1, 29)]

ALL_FEATURES = V_FEATURES + ["Amount"] + EXTRA_FEATURES


def assign_card_ids(df: pd.DataFrame) -> pd.Series:
    """Derive synthetic card IDs from PCA features (same logic as consumer).

    Uses V1-V5 discretised to 3 decimal places, then hashed.
    """
    parts = df[["V1", "V2", "V3", "V4", "V5"]].round(3).astype(str)
    raw = parts.apply(lambda row: "|".join(row), axis=1)
    return raw.apply(lambda x: f"card_{abs(hash(x)) % 10_000:05d}")


def compute_batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all training features on a full DataFrame.

    Args:
        df: Raw dataframe with columns Time, Amount, V1..V28, Class.

    Returns:
        DataFrame with original + engineered features, ready for training.
    """
    df = df.copy().sort_values("Time").reset_index(drop=True)
    df["card_id"] = assign_card_ids(df)

    # ── Per-card rolling features ───────────────────────────────
    # Group by card and compute rolling statistics
    grouped = df.groupby("card_id")

    # Amount z-score (rolling window of 200, matching consumer)
    rolling_mean = grouped["Amount"].transform(
        lambda x: x.rolling(window=200, min_periods=2).mean().shift(1)
    )
    rolling_std = grouped["Amount"].transform(
        lambda x: x.rolling(window=200, min_periods=2).std().shift(1)
    )
    df["amount_zscore"] = np.where(
        rolling_std > 0,
        (df["Amount"] - rolling_mean) / rolling_std,
        0.0,
    )

    # Time since last transaction (per card)
    df["time_since_last_tx"] = grouped["Time"].diff().fillna(0.0)

    # Transaction count in last 60 seconds (per card)
    # Count how many previous txs of same card have Time within [current_time - 60, current_time)
    def _count_recent(times: pd.Series) -> pd.Series:
        counts = []
        time_vals = times.values
        for i in range(len(time_vals)):
            t = time_vals[i]
            window_start = t - 60.0
            cnt = np.sum((time_vals[:i] >= window_start) & (time_vals[:i] < t))
            counts.append(cnt)
        return pd.Series(counts, index=times.index)

    df["tx_count_recent"] = grouped["Time"].transform(_count_recent)

    # Amount relative to rolling mean (min_periods=1 to match streaming)
    rolling_mean_ratio = grouped["Amount"].transform(
        lambda x: x.rolling(window=200, min_periods=1).mean().shift(1)
    )
    df["amount_to_mean"] = np.where(
        rolling_mean_ratio > 0,
        df["Amount"] / rolling_mean_ratio,
        1.0,
    )

    # Amount relative to rolling max
    rolling_max = grouped["Amount"].transform(
        lambda x: x.expanding().max().shift(1)
    )
    df["amount_max_ratio"] = np.where(
        rolling_max > 0,
        df["Amount"] / rolling_max,
        1.0,
    )

    # Fill NaN from first-row-per-card
    for col in EXTRA_FEATURES:
        df[col] = df[col].fillna(0.0)

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for model training.

    X contains V1..V28 + Amount + engineered features.
    y is the Class column.
    """
    enriched = compute_batch_features(df)
    X = enriched[ALL_FEATURES].copy()
    # Rename Amount to lowercase to match consumer's feature vector
    X = X.rename(columns={"Amount": "amount"})
    y = enriched["Class"]
    return X, y
