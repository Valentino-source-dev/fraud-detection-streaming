"""Tests for the batch training feature engineering."""

import numpy as np
import pandas as pd
import pytest

from training.features import (
    ALL_FEATURES,
    EXTRA_FEATURES,
    assign_card_ids,
    compute_batch_features,
    get_feature_matrix,
)


def _make_dataframe(n: int = 100) -> pd.DataFrame:
    """Create a minimal test DataFrame mimicking creditcard.csv."""
    rng = np.random.default_rng(42)
    data = {"Time": np.cumsum(rng.exponential(1.0, n))}
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    data["Amount"] = rng.exponential(50.0, n)
    data["Class"] = rng.choice([0, 1], n, p=[0.98, 0.02])
    return pd.DataFrame(data)


class TestAssignCardIds:
    def test_returns_string_series(self):
        df = _make_dataframe()
        ids = assign_card_ids(df)
        assert isinstance(ids, pd.Series)
        assert ids.dtype == object
        assert all(isinstance(x, str) for x in ids)

    def test_ids_start_with_card(self):
        df = _make_dataframe()
        ids = assign_card_ids(df)
        assert all(x.startswith("card_") for x in ids)


class TestComputeBatchFeatures:
    def test_adds_extra_columns(self):
        df = _make_dataframe()
        result = compute_batch_features(df)
        for col in EXTRA_FEATURES:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans_in_features(self):
        df = _make_dataframe()
        result = compute_batch_features(df)
        for col in EXTRA_FEATURES:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_zscore_reasonable_range(self):
        df = _make_dataframe(500)
        result = compute_batch_features(df)
        # Most z-scores should be within [-5, 5]
        zscores = result["amount_zscore"]
        in_range = ((zscores >= -10) & (zscores <= 10)).mean()
        assert in_range > 0.95

    def test_time_since_last_non_negative(self):
        df = _make_dataframe()
        result = compute_batch_features(df)
        assert (result["time_since_last_tx"] >= 0).all()


class TestGetFeatureMatrix:
    def test_output_shapes(self):
        df = _make_dataframe()
        X, y = get_feature_matrix(df)
        assert len(X) == len(df)
        assert len(y) == len(df)
        assert X.shape[1] == len(ALL_FEATURES)

    def test_column_names_lowercase_amount(self):
        df = _make_dataframe()
        X, _ = get_feature_matrix(df)
        assert "amount" in X.columns
        assert "Amount" not in X.columns

    def test_y_is_binary(self):
        df = _make_dataframe()
        _, y = get_feature_matrix(df)
        assert set(y.unique()).issubset({0, 1})
