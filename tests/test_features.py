"""Tests for src/features.py: the balance-discrepancy features and the
cost-sensitive threshold.

Two things in this repo carry the result, and both are here:

1. The balance-discrepancy features. The README says they carry most of the
   predictive signal, and they are derived from an accounting identity rather
   than learned. An identity either holds or it does not, so it can be asserted
   exactly rather than approximately.

2. pick_best_threshold. This is where the business decision lives: missed fraud
   costs 1000, a false alarm costs 10, a ratio of 100 to 1. A model that ranks
   perfectly and cuts at the wrong threshold still loses money, so the threshold
   logic deserves tests more than the model does.

All tests run on small hand-built frames. None needs the 470 MB PaySim file.
"""
import numpy as np
import pandas as pd
import pytest

from features import (
    ACTIVE_TYPES,
    COST_PER_FALSE_ALARM,
    COST_PER_MISSED_FRAUD,
    FEATURE_COLS,
    engineer_features,
    pick_best_threshold,
)


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
    )


# --------------------------------------------------------------------------
# Balance discrepancy: the accounting identity
# --------------------------------------------------------------------------

def test_clean_transaction_has_zero_discrepancy():
    """newbalanceOrig = oldbalanceOrg - amount, and the destination mirrors it.
    A transaction that obeys both sides of the identity must score zero on both
    discrepancy features, or the signal is measuring something else."""
    df = _frame([[100.0, 500.0, 400.0, 1000.0, 1100.0]])
    out = engineer_features(df)
    assert out["orig_balance_discrepancy"].iloc[0] == pytest.approx(0.0)
    assert out["dest_balance_discrepancy"].iloc[0] == pytest.approx(0.0)


def test_origin_discrepancy_detects_drained_account():
    """The classic PaySim fraud shape: the money leaves but the origin balance is
    reported as unchanged. The identity then breaks by exactly the amount."""
    df = _frame([[100.0, 500.0, 500.0, 0.0, 0.0]])
    out = engineer_features(df)
    assert out["orig_balance_discrepancy"].iloc[0] == pytest.approx(-100.0)


def test_destination_discrepancy_detects_unrecorded_credit():
    """Money arrives but the destination balance does not move."""
    df = _frame([[250.0, 1000.0, 750.0, 400.0, 400.0]])
    out = engineer_features(df)
    assert out["dest_balance_discrepancy"].iloc[0] == pytest.approx(250.0)


def test_drain_ratio_flags_a_full_sweep():
    """Emptying an account gives a drain ratio of 1. This is the feature that
    separates 'a large payment' from 'everything you had'."""
    df = _frame([[500.0, 500.0, 0.0, 0.0, 500.0]])
    out = engineer_features(df)
    assert out["orig_drain_ratio"].iloc[0] == pytest.approx(1.0, rel=1e-6)


def test_ratios_survive_zero_balances():
    """A zero starting balance must not produce inf or NaN, or XGBoost silently
    treats the row differently from every other row."""
    df = _frame([[100.0, 0.0, 0.0, 0.0, 100.0]])
    out = engineer_features(df)
    assert np.isfinite(out["orig_drain_ratio"].iloc[0])
    assert np.isfinite(out["dest_amount_ratio"].iloc[0])


def test_engineer_features_is_not_destructive():
    """It must return a copy. Mutating the caller's frame in place is how a
    pipeline ends up training on features it also leaked into validation."""
    df = _frame([[100.0, 500.0, 400.0, 1000.0, 1100.0]])
    before = df.copy()
    engineer_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_the_five_model_features_are_all_listed():
    """Pin FEATURE_COLS membership by name.

    The test below iterates FEATURE_COLS to check each one is produced, which
    means deleting an entry deletes its own check: the suite stays green while
    the model quietly trains on fewer features. This names them instead.

    The set is deliberately small. The raw balance columns were removed because
    the model was using "balance hits zero" as a shortcut, which also fires on a
    perfectly ordinary account closure, so only `amount` and the four derived
    discrepancy and ratio features remain.
    """
    assert set(FEATURE_COLS) == {
        "amount",
        "orig_balance_discrepancy",
        "dest_balance_discrepancy",
        "orig_drain_ratio",
        "dest_amount_ratio",
    }


def test_raw_balance_columns_are_not_model_features():
    """The shortcut that the feature set exists to prevent. If a raw balance
    column reappears in FEATURE_COLS, the model can learn "drained to zero"
    again instead of whether the accounting identity broke."""
    raw = {"oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
    assert raw.isdisjoint(set(FEATURE_COLS))


def test_all_declared_feature_columns_are_produced():
    """FEATURE_COLS is what the model is trained on. If engineer_features stops
    producing one of them, training fails far from the cause."""
    df = _frame([[100.0, 500.0, 400.0, 1000.0, 1100.0]])
    out = engineer_features(df)
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    assert not missing, f"missing engineered columns: {missing}"


def test_active_types_exclude_types_with_no_fraud():
    """PaySim fraud only ever occurs in TRANSFER and CASH_OUT. Filtering to those
    is what turns 6.36M rows into 2.77M without losing a single positive."""
    assert set(ACTIVE_TYPES) == {"TRANSFER", "CASH_OUT"}


# --------------------------------------------------------------------------
# Cost-sensitive threshold: where the business decision lives
# --------------------------------------------------------------------------

def test_cost_asymmetry_is_one_hundred_to_one():
    """If these constants change, every threshold in the repo changes with them,
    so the ratio is pinned deliberately."""
    assert COST_PER_MISSED_FRAUD / COST_PER_FALSE_ALARM == 100


def _total_cost(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    missed = int(((pred == 0) & (y_true == 1)).sum())
    false_alarms = int(((pred == 1) & (y_true == 0)).sum())
    return COST_PER_MISSED_FRAUD * missed + COST_PER_FALSE_ALARM * false_alarms


def test_chosen_threshold_is_cost_optimal():
    """The property that actually matters: no other cutoff can beat the one
    returned.

    The sweep below is a dense grid built here, deliberately not
    `np.unique(np.concatenate([probs, [0.0, 1.0]]))`, which is the exact
    candidate set the implementation itself builds. Reusing that set would
    inherit the search space from the code under test, so a wrong candidate set
    would pass. An independent grid catches that as well as an inverted
    comparison.
    """
    rng = np.random.default_rng(42)
    n = 600
    y = rng.binomial(1, 0.05, size=n)
    probs = np.clip(rng.normal(loc=y * 0.6 + 0.2, scale=0.2), 0, 1)

    best = pick_best_threshold(y, probs)
    best_cost = _total_cost(y, probs, best)
    for candidate in np.linspace(0.0, 1.0, 1001):
        assert _total_cost(y, probs, candidate) >= best_cost - 1e-9


def test_threshold_is_low_because_misses_are_expensive():
    """With missed fraud 100x costlier than a false alarm, the optimiser should
    land well below 0.5. A threshold near 0.5 means the cost weighting is not
    reaching the decision."""
    rng = np.random.default_rng(7)
    n = 800
    y = rng.binomial(1, 0.03, size=n)
    probs = np.clip(rng.normal(loc=y * 0.5 + 0.15, scale=0.2), 0, 1)
    assert pick_best_threshold(y, probs) < 0.5


def test_perfectly_separable_threshold_catches_all_fraud():
    """When the classes are cleanly separated, the optimal decision is to catch
    every fraud at zero false alarms."""
    y = np.array([0, 0, 0, 0, 1, 1])
    probs = np.array([0.01, 0.02, 0.03, 0.04, 0.90, 0.95])
    t = pick_best_threshold(y, probs)
    pred = (probs >= t).astype(int)
    assert (pred == y).all()


def test_threshold_is_within_unit_interval():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.1, size=300)
    probs = rng.random(300)
    assert 0.0 <= pick_best_threshold(y, probs) <= 1.0
