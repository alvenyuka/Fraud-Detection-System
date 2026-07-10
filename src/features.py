"""
features.py — shared feature engineering for the Fraud Detection System.

Single source of truth for the PaySim loading/filtering and balance-discrepancy
feature engineering used by train.py, predict.py, tune.py, validate.py, and
monitoring.py. Keeping this in one place means every script that scores or
trains on PaySim data does it identically.
"""

import logging

import numpy as np
import pandas as pd

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
# the raw balances forces the model to rely only on whether the accounting
# identity actually breaks, which is the real fraud signal.
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

    eps = 1e-8
    df["orig_drain_ratio"] = df["amount"] / (df["oldbalanceOrg"] + eps)
    df["dest_amount_ratio"] = df["amount"] / (df["oldbalanceDest"] + eps)

    return df


def pick_best_threshold(y_true: np.ndarray, predicted_probs: np.ndarray) -> float:
    """
    Try every possible cutoff and keep the one with the lowest total cost.

    A low threshold catches more fraud but raises more false alarms; a high
    threshold does the opposite. This picks the balance point using the
    $1000 (missed fraud) vs $10 (false alarm) costs above. Used by both
    train.py (to pick the threshold the shipped model ships with) and
    validate.py (to pick each walk-forward fold's own threshold) -- a
    threshold tuned for one set of hyperparameters/features isn't
    automatically right for another, so this always gets recomputed rather
    than hardcoded.
    """
    candidate_thresholds = np.unique(np.concatenate([predicted_probs, [0.0, 1.0]]))
    best_threshold, lowest_cost = 0.5, np.inf

    for threshold in candidate_thresholds:
        predicted_fraud = (predicted_probs >= threshold).astype(int)
        missed_fraud_count = int(((predicted_fraud == 0) & (y_true == 1)).sum())
        false_alarm_count = int(((predicted_fraud == 1) & (y_true == 0)).sum())
        total_cost = COST_PER_MISSED_FRAUD * missed_fraud_count + COST_PER_FALSE_ALARM * false_alarm_count
        if total_cost < lowest_cost:
            lowest_cost, best_threshold = total_cost, threshold

    return float(best_threshold)
