"""
train.py — Step 1 of the model build-up: the baseline model.

This is where the project starts: load PaySim data, engineer the
balance-discrepancy features, train an XGBoost classifier, and calibrate its
probabilities so a "0.9" score really does mean roughly a 90% chance of fraud.

The rest of the build-up lives in separate scripts, each answering a
question this baseline leaves open:
  - src/tune.py     — were these hyperparameters ever tested against alternatives?
  - src/validate.py — does this hold up on more than one train/test split?
  - src/monitoring.py — how would we know if the model started drifting?
  - dashboard/app.py  — how does someone without Python actually use this?

Usage
-----
    python src/train.py --data PS_20174392719_1491204439457_log.csv

    # or with explicit output path:
    python src/train.py --data data/paysim.csv --model-out model/xgb_fraud_model.pkl
"""

import argparse
import json
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

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, engineer_features, load_and_filter, pick_best_threshold  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPLIT_STEP = 490  # ~66% of 744-step horizon
BEST_PARAMS_PATH = Path(__file__).parent.parent / "model" / "best_params.json"


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

DEFAULT_XGB_PARAMS = {
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

def load_xgb_params() -> dict:
    """Use tuned hyperparameters from src/tune.py if available, else the defaults."""
    if BEST_PARAMS_PATH.exists():
        tuned = json.loads(BEST_PARAMS_PATH.read_text())
        params = {**DEFAULT_XGB_PARAMS, **tuned}
        log.info("Using tuned hyperparameters from %s", BEST_PARAMS_PATH)
        return params
    log.info("No %s found — using default hyperparameters.", BEST_PARAMS_PATH.name)
    return DEFAULT_XGB_PARAMS


def train(data_path: str, model_out: str) -> None:
    xgb_params = load_xgb_params()
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
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_fit, y_fit)

    log.info("Calibrating probabilities (isotonic) on %d held-out rows ...", len(X_calib))
    calibrated = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
    calibrated.fit(X_calib, y_calib)

    # Pick the decision threshold from the calibration split, not the test
    # split -- choosing it on the same rows we report metrics on would let
    # the threshold overfit to the number we're trying to honestly report.
    # A threshold tuned for one set of hyperparameters/features isn't
    # automatically right for another (see MODEL_CARD.md), so this is
    # recomputed every training run rather than hardcoded.
    calib_probs = calibrated.predict_proba(X_calib)[:, 1]
    operating_threshold = pick_best_threshold(y_calib.to_numpy(), calib_probs)
    log.info("Chosen decision threshold (cost-optimal on calibration split): %.4f", operating_threshold)

    probs = calibrated.predict_proba(X_test)[:, 1]
    preds = (probs >= operating_threshold).astype(int)

    metrics = {
        "PR-AUC":    round(average_precision_score(y_test, probs), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, probs), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall":    round(recall_score(y_test, preds), 4),
        "F1":        round(f1_score(y_test, preds), 4),
        "Brier":     round(brier_score_loss(y_test, probs), 6),
        "Threshold": round(operating_threshold, 4),
    }

    log.info("Test-set metrics:")
    for k, v in metrics.items():
        log.info("  %-12s %s", k, v)

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model":               calibrated,
        "feature_cols":        FEATURE_COLS,
        "operating_threshold": operating_threshold,
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
