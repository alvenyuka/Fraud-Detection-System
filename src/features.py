"""
features.py: shared feature engineering for the Fraud Detection System.

Single source of truth for the PaySim loading/filtering and balance-discrepancy
feature engineering used by train.py, predict.py, tune.py, validate.py, and
monitoring.py. Keeping this in one place means every script that scores or
trains on PaySim data does it identically.
"""

import logging

import numpy as np
import pandas as pd

from config import HOLDOUT_STEP_SHARE

log = logging.getLogger(__name__)

# Same cost framework used throughout this project: a missed fraud (false
# negative) costs far more than a false alarm (false positive).
COST_PER_MISSED_FRAUD = 1000
COST_PER_FALSE_ALARM = 10

ACTIVE_TYPES = {"TRANSFER", "CASH_OUT"}

# The raw balance columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest,
# newbalanceDest) used to be model inputs alongside the engineered features
# below. Testing found that let the model take a shortcut: PaySim's simulated
# fraud almost always drains the sender's account to exactly zero, so the
# model learned "balance hits zero" as a fraud signal on its own -- even for
# a transaction with a perfectly consistent, zero-discrepancy destination
# update (e.g. closing an account, or moving your whole balance somewhere
# else, which are completely normal, non-fraudulent things to do). Dropping
# the raw balances was meant to make the model rely only on whether the
# accounting identity actually breaks. It did so only in part:
# orig_drain_ratio still encodes a full drain, and the shipped model still
# flags a fully consistent full drain (dashboard/data/scenario_table.json,
# written by src/scenarios.py). MODEL_CARD.md explains why.
FEATURE_COLS = [
    "amount",
    "orig_balance_discrepancy",
    "dest_balance_discrepancy",
    "orig_drain_ratio",
    "dest_amount_ratio",
]


def load_and_filter(path: str) -> pd.DataFrame:
    """Load PaySim CSV and keep only fraud-active transaction types."""
    log.info("Loading %s ...", path)
    df = pd.read_csv(path)
    n_loaded = len(df)
    log.info("Loaded %d rows", n_loaded)
    df = df[df["type"].isin(ACTIVE_TYPES)].copy()
    log.info("After type filter: %d rows (%.1f%% kept)", len(df), 100 * len(df) / max(n_loaded, 1))
    return df


def split_off_latest_steps(train_df: pd.DataFrame, share: float = HOLDOUT_STEP_SHARE):
    """
    Split a training period into a fit part and a later calibration part by time.

    The calibration part is the latest `share` of the period's steps, so the
    isotonic calibrator and the decision threshold are fitted on the rows
    closest in time to the ones the model will score, as src/tune.py already
    does for early stopping. A random slice would interleave calibration rows
    with fit rows and inherit the early period's much lower fraud prevalence.

    Returns (fit_df, calibration_df). Raises ValueError if the calibration part
    holds no fraud, because a threshold cannot be chosen from it.
    """
    first, last = int(train_df["step"].min()), int(train_df["step"].max())
    cut = last - int((last - first + 1) * share)
    fit_df = train_df[train_df["step"] <= cut]
    calibration_df = train_df[train_df["step"] > cut]
    if calibration_df["isFraud"].sum() == 0 or fit_df["isFraud"].sum() == 0:
        raise ValueError(f"steps {first}-{last} split at {cut} leave one side without fraud")
    return fit_df, calibration_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the four engineered features the model is built on.

    dashboard/data/feature_importance.json, written by src/explain.py, gives
    each feature's share of mean |SHAP| attribution for the shipped model.

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

    eps = 1e-8
    df["orig_drain_ratio"] = df["amount"] / (df["oldbalanceOrg"] + eps)
    df["dest_amount_ratio"] = df["amount"] / (df["oldbalanceDest"] + eps)

    return df


def pick_best_threshold(y_true: np.ndarray, predicted_probs: np.ndarray) -> float:
    """
    Try every possible cutoff and keep the one with the lowest total cost.

    A low threshold catches more fraud but raises more false alarms; a high
    threshold does the opposite. This picks the balance point using the
    1,000 (missed fraud) vs 10 (false alarm) cost units above. Used by both
    train.py (to pick the threshold the shipped model ships with) and
    validate.py (to pick each walk-forward fold's own threshold), because a
    threshold tuned for one set of hyperparameters/features isn't
    automatically right for another, so this always gets recomputed rather
    than hardcoded.

    Both callers must hand this the calibration split's labels, never the test
    split's. Choosing the cutoff on the same rows the metrics are reported on
    makes those metrics oracle-thresholded rather than out-of-sample.

    On a cost tie this returns the highest of the tied thresholds: the
    candidates are ascending and a later candidate replaces the incumbent when
    its cost is equal or lower. At equal cost the higher threshold is the better
    operational choice, since it raises fewer alerts for the same money.
    """
    candidate_thresholds = np.unique(np.concatenate([predicted_probs, [0.0, 1.0]]))
    best_threshold, lowest_cost = 0.5, np.inf

    for threshold in candidate_thresholds:
        predicted_fraud = (predicted_probs >= threshold).astype(int)
        missed_fraud_count = int(((predicted_fraud == 0) & (y_true == 1)).sum())
        false_alarm_count = int(((predicted_fraud == 1) & (y_true == 0)).sum())
        total_cost = COST_PER_MISSED_FRAUD * missed_fraud_count + COST_PER_FALSE_ALARM * false_alarm_count
        if total_cost <= lowest_cost:
            lowest_cost, best_threshold = total_cost, threshold

    return float(best_threshold)
