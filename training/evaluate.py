"""Evaluate the production model with proper imbalanced-class metrics.

Produces:
  - AUC-PR, AUC-ROC, Precision, Recall, F1 (fraud class)
  - Confusion matrix
  - Precision-Recall curve
  - SHAP summary plot (top 20 features)
  - All artifacts logged to MLflow

Usage:
    cd training && python evaluate.py
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import training_config as config
from features import get_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate")


def main() -> None:
    # ── Load data ────────────────────────────────────────────
    raw = pd.read_csv(config.DATA_PATH)
    X, y = get_feature_matrix(raw)
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    log.info("Test set: %d rows (%d fraud).", len(X_test), y_test.sum())

    # ── Load production model ────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{config.MLFLOW_MODEL_NAME}@production"
    log.info("Loading model from %s ...", model_uri)
    model = mlflow.xgboost.load_model(model_uri)

    # ── Predictions ──────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # ── Metrics ──────────────────────────────────────────────
    metrics = {
        "auc_pr": average_precision_score(y_test, y_proba),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "precision_fraud": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_fraud": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_fraud": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }

    log.info("=== Evaluation Results ===")
    for name, value in metrics.items():
        log.info("  %-20s %.4f", name, value)

    log.info("\n%s", classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # ── Log to MLflow ────────────────────────────────────────
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)

        # Confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Legit", "Fraud"],
            cmap="Blues",
            ax=ax_cm,
        )
        ax_cm.set_title("Confusion Matrix")
        plt.tight_layout()
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        # Precision-Recall curve
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay.from_predictions(
            y_test, y_proba,
            name="XGBoost",
            ax=ax_pr,
        )
        ax_pr.set_title(f"Precision-Recall Curve (AUC-PR = {metrics['auc_pr']:.4f})")
        plt.tight_layout()
        mlflow.log_figure(fig_pr, "precision_recall_curve.png")
        plt.close(fig_pr)

        # SHAP summary plot
        log.info("Computing SHAP values (this may take a minute)...")
        try:
            explainer = shap.TreeExplainer(model)
            # Use a sample for speed
            sample_size = min(2000, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=config.RANDOM_SEED)
            shap_values = explainer.shap_values(X_sample)

            fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_sample,
                max_display=20,
                show=False,
            )
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "shap_summary.png")
            plt.close("all")
            log.info("SHAP summary plot logged.")
        except Exception as exc:
            log.warning("SHAP computation failed: %s", exc)

    log.info("Evaluation complete. Artifacts logged to MLflow.")


if __name__ == "__main__":
    main()
