"""
train.py — Fraud Detection System training pipeline

Trains an XGBoost classifier on PaySim mobile-money data with:
  - Time-based train/test split at step 490
  - Balance-discrepancy feature engineering
  - Calibrated probability output
  - Model serialised to model/xgb_fraud_model.pkl

Usage
-----
    python src/train.py --data PS_20174392719_1491204439457_log.csv

    # or with explicit output path:
    python src/train.py --data data/paysim.csv --model-out model/xgb_fraud_model.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

ACTIVE_TYPES = {"TRANSFER", "CASH_OUT"}
SPLIT_STEP = 490  # ~66% of 744-step horizon


def load_and_filter(path: str) -> pd.DataFrame:
    """Load PaySim CSV and keep only fraud-active transaction types."""
    log.info("Loading %s …", path)
    df = pd.read_csv(path)
    log.info("Loaded %d rows", len(df))
    df = df[df["type"].isin(ACTIVE_TYPES)].copy()
    log.info("After type filter: %d rows (%.1f%% kept)", len(df), 100 * len(df) / 6_362_620)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add balance-discrepancy features that carry ~85% of predictive signal.

    Accounting identity for a clean transaction:
        newbalanceOrig  = oldbalanceOrg  - amount
        newbalanceDest  = oldbalanceDest + amount

    Any deviation flags a potential fraud.
    """
    df = df.copy()

    df["orig_balance_discrepancy"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["dest_balance_discrepancy"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    # Ratio features (epsilon avoids div-by-zero)
    eps = 1e-8
    df["orig_drain_ratio"] = df["amount"] / (df["oldbalanceOrg"] + eps)
    df["dest_amount_ratio"] = df["amount"] / (df["oldbalanceDest"] + eps)

    return df


FEATURE_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_discrepancy",
    "dest_balance_discrepancy",
    "orig_drain_ratio",
    "dest_amount_ratio",
]


def time_based_split(df: pd.DataFrame, split_step: int = SPLIT_STEP):
    """Split on transaction step to prevent future-state leakage."""
    train = df[df["step"] <= split_step]
    test = df[df["step"] > split_step]
    log.info(
        "Train: %d rows (%.3f%% fraud)  |  Test: %d rows (%.3f%% fraud)",
        len(train), 100 * train["isFraud"].mean(),
        len(test),  100 * test["isFraud"].mean(),
    )
    return train, test


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

OPERATING_THRESHOLD = 0.9989


def train(data_path: str, model_out: str) -> None:
    df = load_and_filter(data_path)
    df = engineer_features(df)

    train_df, test_df = time_based_split(df)

    # Hold out a calibration slice from the training period so isotonic
    # regression isn't fit on rows XGBoost has already memorised — fitting a
    # calibrator on the same data used to train the base model overstates
    # how well-calibrated the model actually is on unseen data.
    fit_df, calib_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["isFraud"], random_state=42
    )

    X_fit,   y_fit   = fit_df[FEATURE_COLS],   fit_df["isFraud"]
    X_calib, y_calib = calib_df[FEATURE_COLS], calib_df["isFraud"]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df["isFraud"]

    log.info("Fitting XGBClassifier on %d rows ...", len(X_fit))
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_fit, y_fit)

    log.info("Calibrating probabilities (isotonic) on %d held-out rows ...", len(X_calib))
    calibrated = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
    calibrated.fit(X_calib, y_calib)

    probs = calibrated.predict_proba(X_test)[:, 1]
    preds = (probs >= OPERATING_THRESHOLD).astype(int)

    metrics = {
        "PR-AUC":    round(average_precision_score(y_test, probs), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, probs), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall":    round(recall_score(y_test, preds), 4),
        "F1":        round(f1_score(y_test, preds), 4),
        "Brier":     round(brier_score_loss(y_test, probs), 6),
        "Threshold": OPERATING_THRESHOLD,
    }

    log.info("Test-set metrics:")
    for k, v in metrics.items():
        log.info("  %-12s %s", k, v)

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model":               calibrated,
        "feature_cols":        FEATURE_COLS,
        "operating_threshold": OPERATING_THRESHOLD,
        "metrics":             metrics,
    }
    joblib.dump(artifact, out_path)
    log.info("Model saved -> %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the PaySim XGBoost fraud classifier."
    )
    p.add_argument(
        "--data",
        required=True,
        metavar="CSV",
        help="Path to PaySim CSV.",
    )
    p.add_argument(
        "--model-out",
        default="model/xgb_fraud_model.pkl",
        metavar="PKL",
        help="Output path for serialised model (default: model/xgb_fraud_model.pkl).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model_out)
