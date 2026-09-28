"""Tests for the time-based split (src/train.py) and PSI drift (src/monitoring.py).

The split is the most important test in this repository. Fraud data is temporal,
and a random split lets the model see later transactions from the same accounts
it is later asked to score. That inflates validation metrics and the inflation
survives every other check, because nothing downstream can tell a leaked score
from an earned one. The only place it can be caught is here.

PSI is what tells you the model has gone stale in production, so it needs to be
zero when nothing moved and to grow monotonically when something does.
"""
import numpy as np
import pandas as pd
import pytest

from monitoring import calculate_psi
from train import SPLIT_STEP, time_based_split


def _stepped_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "step": rng.integers(1, 744, size=n),
            "isFraud": rng.binomial(1, 0.02, size=n),
            "amount": rng.random(n) * 1000,
        }
    )


# --------------------------------------------------------------------------
# Time-based split: the leakage guard
# --------------------------------------------------------------------------

def test_split_produces_no_temporal_overlap():
    """The property that prevents leakage: every training step must be at or
    before the cut, and every test step strictly after it."""
    df = _stepped_frame()
    train, test = time_based_split(df, split_step=400)
    assert train["step"].max() <= 400
    assert test["step"].min() > 400
    assert train["step"].max() < test["step"].min()


def test_split_loses_no_rows():
    df = _stepped_frame()
    train, test = time_based_split(df, split_step=400)
    assert len(train) + len(test) == len(df)


def test_split_is_not_random():
    """Running twice must give identical splits. A split that varies between runs
    cannot be compared across experiments."""
    df = _stepped_frame()
    a_train, a_test = time_based_split(df, split_step=400)
    b_train, b_test = time_based_split(df, split_step=400)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_default_split_step_leaves_both_sides_populated():
    """SPLIT_STEP is ~66% of the 744-step horizon. If it ever drifts past the end
    of the data, the test set silently becomes empty and every metric becomes
    meaningless rather than wrong."""
    assert 0 < SPLIT_STEP < 744
    df = _stepped_frame(n=2000)
    train, test = time_based_split(df)
    assert len(train) > 0 and len(test) > 0


def test_a_random_split_would_overlap():
    """Demonstrates what the guard is guarding against: shuffling the same frame
    puts later steps into training and earlier ones into test."""
    df = _stepped_frame()
    shuffled = df.sample(frac=1.0, random_state=0)
    r_train = shuffled.iloc[: len(df) // 2]
    r_test = shuffled.iloc[len(df) // 2 :]
    assert r_train["step"].max() > r_test["step"].min(), (
        "a random split should overlap in time, which is the whole problem"
    )


# --------------------------------------------------------------------------
# PSI drift
# --------------------------------------------------------------------------

def test_psi_is_zero_for_identical_samples():
    rng = np.random.default_rng(42)
    x = rng.normal(size=4000)
    assert calculate_psi(x, x.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    """PSI is used as a drift threshold, so it has to be monotonic in the shift or
    the threshold means nothing."""
    rng = np.random.default_rng(42)
    ref = rng.normal(size=6000)
    small = calculate_psi(ref, rng.normal(loc=0.25, size=6000))
    large = calculate_psi(ref, rng.normal(loc=1.25, size=6000))
    assert 0 < small < large


def test_psi_returns_zero_for_a_constant_feature():
    """A feature that never varies has no distribution to compare. Returning 0
    rather than raising keeps a single dead column from stopping a monitoring
    run across every other feature."""
    constant = np.full(1000, 7.0)
    assert calculate_psi(constant, constant.copy()) == 0.0


def test_psi_handles_values_outside_the_reference_range():
    """Bin edges are opened to infinity at both ends, so new extreme values land
    in the outer bins instead of being dropped. Dropping them would make a
    genuinely drifting feature look stable."""
    rng = np.random.default_rng(42)
    ref = rng.normal(size=3000)
    shifted = rng.normal(loc=8.0, size=3000)  # entirely outside the reference
    assert calculate_psi(ref, shifted) > 0.5


def test_psi_is_non_negative():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=2000)
    for loc in (-1.0, 0.0, 1.0, 3.0):
        assert calculate_psi(ref, rng.normal(loc=loc, size=2000)) >= 0.0
