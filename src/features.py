"""
features.py — shared feature engineering for the Fraud Detection System.

Single source of truth for the PaySim loading/filtering and balance-discrepancy
feature engineering used by train.py, predict.py, tune.py, validate.py, and
monitoring.py. Keeping this in one place means every script that scores or
trains on PaySim data does it identically.
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

ACTIVE_TYPES = {"TRANSFER", "CASH_OUT"}

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
