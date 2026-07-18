"""Train an XGBoost fraud detection model with Optuna hyperparameter tuning.

Usage:
    # With MLflow running (e.g. via docker compose up mlflow postgres):
    cd training && python train.py

    # Or specify MLflow URI:
    MLFLOW_TRACKING_URI=http://localhost:5000 python train.py

Logs everything to MLflow: parameters, metrics, model artifact, feature
importance plot.  Registers the best model in the MLflow Model Registry.
"""

from __future__ import annotations

import logging
import sys

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import training_config as config
from features import get_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load CSV, compute features, split into train/test."""
    log.info("Loading dataset from %s ...", config.DATA_PATH)
    raw = pd.read_csv(config.DATA_PATH)
    log.info("Raw dataset: %d rows, %d frauds (%.3f%%).",
             len(raw), raw["Class"].sum(), raw["Class"].mean() * 100)

    X, y = get_feature_matrix(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    log.info("Train: %d (%d fraud) | Test: %d (%d fraud).",
             len(X_train), y_train.sum(), len(X_test), y_test.sum())
    return X_train, X_test, y_train, y_test


def _objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    """Optuna objective: maximise AUC-PR via 3-fold stratified CV."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 600.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_SEED)
    aucprs = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=config.RANDOM_SEED,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(
            Xt, yt,
            eval_set=[(Xv, yv)],
            verbose=False,
        )
        y_proba = model.predict_proba(Xv)[:, 1]
        aucprs.append(average_precision_score(yv, y_proba))

    return float(np.mean(aucprs))


def _train_final_model(
    params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Train the final model with best params and log everything to MLflow."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-best") as run:
        log.info("MLflow run: %s", run.info.run_id)

        model = XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=config.RANDOM_SEED,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Predictions ──────────────────────────────────
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # ── Metrics (fraud class only) ───────────────────
        metrics = {
            "auc_pr": average_precision_score(y_test, y_proba),
            "auc_roc": roc_auc_score(y_test, y_proba),
            "precision_fraud": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall_fraud": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "f1_fraud": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        }
        for name, value in metrics.items():
            log.info("  %s = %.4f", name, value)

        # ── Log to MLflow ────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("test_size", config.TEST_SIZE)
        mlflow.log_param("random_seed", config.RANDOM_SEED)

        # Log model
        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.MLFLOW_MODEL_NAME,
        )
        log.info("Model logged and registered as '%s'.", config.MLFLOW_MODEL_NAME)

        # ── Feature importance plot ──────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            importance = model.feature_importances_
            feature_names = X_train.columns.tolist()
            sorted_idx = np.argsort(importance)[-20:]  # top 20

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                [feature_names[i] for i in sorted_idx],
                importance[sorted_idx],
            )
            ax.set_title("Top 20 Feature Importances")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)
            log.info("Feature importance plot logged.")
        except Exception as exc:
            log.warning("Could not generate feature importance plot: %s", exc)

        # ── Promote to Production (using modern Aliases) ──
        try:
            client = mlflow.tracking.MlflowClient()
            version = getattr(model_info, "registered_model_version", None)
            if not version:
                versions = client.search_model_versions(f"name='{config.MLFLOW_MODEL_NAME}'")
                version = versions[0].version if versions else "1"
            
            client.set_registered_model_alias(
                name=config.MLFLOW_MODEL_NAME,
                alias="production",
                version=version,
            )
            log.info("Model version %s assigned alias 'production'.", version)
        except Exception as exc:
            log.warning("Could not assign 'production' alias: %s", exc)



def main() -> None:
    """Full training pipeline: load → tune → train → register."""
    X_train, X_test, y_train, y_test = _load_data()

    # ── Hyperparameter tuning ────────────────────────
    log.info("Starting Optuna hyperparameter tuning (%d trials)...", config.OPTUNA_TRIALS)
    study = optuna.create_study(direction="maximize", study_name="xgboost-fraud")
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train),
        n_trials=config.OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    log.info("Best trial AUC-PR: %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)

    # ── Train final model ────────────────────────────
    _train_final_model(study.best_params, X_train, X_test, y_train, y_test)
    log.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
