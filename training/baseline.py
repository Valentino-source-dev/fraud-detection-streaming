"""Isolation Forest baseline — unsupervised anomaly detection for thesis comparison.

This is NOT deployed in the live pipeline.  It trains an Isolation Forest on
the same features and evaluates it with the same metrics, so the thesis can
discuss: "What happens when you don't have labels?"

Usage:
    cd training && python baseline.py
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import training_config as config
from features import get_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baseline")


def main() -> None:
    # ── Load data ────────────────────────────────────────────
    raw = pd.read_csv(config.DATA_PATH)
    X, y = get_feature_matrix(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    log.info("Train: %d | Test: %d (%d fraud).", len(X_train), len(X_test), y_test.sum())

    # ── Train Isolation Forest ───────────────────────────────
    # contamination set to the approximate fraud rate in the dataset
    contamination = float(y_train.mean())
    log.info("Training Isolation Forest (contamination=%.5f)...", contamination)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_features=1.0,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train)

    # ── Predictions ──────────────────────────────────────────
    # IsolationForest: decision_function returns anomaly scores
    # Lower (more negative) = more anomalous
    raw_scores = model.decision_function(X_test)

    # Normalise to [0, 1] where 1 = most anomalous (for comparison with XGBoost)
    scaler = MinMaxScaler()
    anomaly_scores = 1.0 - scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()

    # IF predict: -1 = anomaly, 1 = normal → convert to 0/1
    y_pred_if = model.predict(X_test)
    y_pred = np.where(y_pred_if == -1, 1, 0)

    # ── Metrics ──────────────────────────────────────────────
    metrics = {
        "auc_pr": average_precision_score(y_test, anomaly_scores),
        "auc_roc": roc_auc_score(y_test, anomaly_scores),
        "precision_fraud": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_fraud": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_fraud": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }

    log.info("=== Isolation Forest Baseline Results ===")
    for name, value in metrics.items():
        log.info("  %-20s %.4f", name, value)

    log.info("\n%s", classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # ── Log to MLflow ────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="isolation-forest-baseline"):
        mlflow.log_params({
            "model_type": "IsolationForest",
            "n_estimators": 200,
            "contamination": contamination,
            "purpose": "baseline_comparison_only",
        })
        mlflow.log_metrics(metrics)

        # Precision-Recall curve
        fig, ax = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay.from_predictions(
            y_test, anomaly_scores,
            name="Isolation Forest",
            ax=ax,
        )
        ax.set_title(f"Isolation Forest — PR Curve (AUC-PR = {metrics['auc_pr']:.4f})")
        plt.tight_layout()
        mlflow.log_figure(fig, "baseline_pr_curve.png")
        plt.close(fig)

        # Score distribution
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(anomaly_scores[y_test == 0], bins=50, alpha=0.6, label="Legit", color="green")
        ax2.hist(anomaly_scores[y_test == 1], bins=50, alpha=0.6, label="Fraud", color="red")
        ax2.set_xlabel("Anomaly Score")
        ax2.set_ylabel("Count")
        ax2.set_title("Anomaly Score Distribution")
        ax2.legend()
        plt.tight_layout()
        mlflow.log_figure(fig2, "baseline_score_distribution.png")
        plt.close(fig2)

    log.info("Baseline evaluation complete. Artifacts logged to MLflow.")


if __name__ == "__main__":
    main()
