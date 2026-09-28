"""Parity between api/score.py's pure-Python port and the shipped model.

The live demo does not call scikit-learn or xgboost. It re-implements tree
traversal and isotonic calibration in stdlib Python against
model/model_export.json, which makes it the single component in this repo most
able to break silently: a wrong default-branch rule or an off-by-one in the
isotonic scan changes scores without raising anything, and the demo would keep
serving plausible numbers.

Two prose claims used to stand in for this test, one in api/score.py and one in
src/export_model_json.py, quoting different row counts and different bounds,
with the evidence living outside the repo. This replaces both with a bound that
is measured here on every run.

Neither the PaySim CSV nor a training run is needed: both artifacts
(model/xgb_fraud_model.pkl and model/model_export.json) are committed, and the
transactions are generated from a fixed seed.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PICKLE_PATH = REPO_ROOT / "model" / "xgb_fraud_model.pkl"
EXPORT_PATH = REPO_ROOT / "model" / "model_export.json"

# api/score.py fetches the export over HTTP at import time unless this is set.
# It has to be set before the import below, not inside a fixture.
os.environ["FRAUD_MODEL_EXPORT_PATH"] = str(EXPORT_PATH)

pytestmark = pytest.mark.skipif(
    not (PICKLE_PATH.exists() and EXPORT_PATH.exists()),
    reason="needs the committed model artifacts",
)

N_ROWS = 20_000
SEED = 42

# Measured, not assumed. On this fixture the worst row disagrees by about
# 1.9e-06 and the mean by about 6e-10. The bounds below sit a little above
# both, so ordinary float noise passes and a real divergence in the traversal
# or the calibration scan does not.
MAX_ABS_DIFF = 1e-5
MAX_MEAN_DIFF = 1e-8


@pytest.fixture(scope="module")
def transactions():
    """A fixed-seed spread of transactions, including the shapes that matter.

    Roughly a third drain the sender to zero and roughly a third leave the
    recipient's balance untouched, because those are the two cases the model
    keys on and the two places the port could diverge without anyone noticing.
    """
    rng = np.random.default_rng(SEED)
    old_orig = rng.lognormal(9, 1.5, N_ROWS)
    amount = old_orig * rng.random(N_ROWS)
    old_dest = rng.lognormal(8, 2, N_ROWS)
    new_orig = np.where(rng.random(N_ROWS) < 0.3, 0.0, old_orig - amount)
    new_dest = np.where(rng.random(N_ROWS) < 0.3, old_dest, old_dest + amount)
    return pd.DataFrame(
        {
            "amount": amount,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
        }
    )


@pytest.fixture(scope="module")
def differences(transactions):
    import sys

    sys.path.insert(0, str(REPO_ROOT / "api"))
    import score  # noqa: PLC0415 -- deliberately imported after the env var is set

    from features import FEATURE_COLS, engineer_features  # noqa: PLC0415

    bundle = joblib.load(PICKLE_PATH)
    reference = bundle["model"].predict_proba(
        engineer_features(transactions)[FEATURE_COLS]
    )[:, 1]

    ported = np.array(
        [
            score.score_transaction(row)["fraud_probability"]
            for row in transactions.to_dict("records")
        ]
    )
    return np.abs(ported - reference)


def test_worst_row_agrees_to_the_measured_bound(differences):
    assert differences.max() < MAX_ABS_DIFF


def test_typical_row_agrees_far_more_closely(differences):
    """The worst case is a handful of rows on a split threshold. The bulk of
    the sample has to agree several orders of magnitude tighter than that, or
    the port is wrong in a way a max alone would not show."""
    assert differences.mean() < MAX_MEAN_DIFF
    assert np.percentile(differences, 99) < MAX_MEAN_DIFF


def test_the_edge_cases_the_model_keys_on_agree(transactions):
    """A full drain, a zero-balance sender and a tiny amount, which are the
    three shapes the score is most sensitive to.

    Worth knowing rather than glossing: the zero-balance and tiny-amount rows
    match bit for bit, but the full drain does not. It disagrees by about
    3.7e-06, the largest gap found anywhere, because a full drain puts the raw
    margin close to an isotonic breakpoint and the two implementations
    interpolate the same segment in a different float order. It is noise, not
    a modelling difference, and calling it an exact match would be wrong.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "api"))
    import score  # noqa: PLC0415

    from features import FEATURE_COLS, engineer_features  # noqa: PLC0415

    edge = pd.DataFrame(
        [
            {"amount": 10_000.0, "oldbalanceOrg": 10_000.0, "newbalanceOrig": 0.0,
             "oldbalanceDest": 2_000.0, "newbalanceDest": 2_000.0},
            {"amount": 500.0, "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
             "oldbalanceDest": 0.0, "newbalanceDest": 500.0},
            {"amount": 0.01, "oldbalanceOrg": 1_000.0, "newbalanceOrig": 999.99,
             "oldbalanceDest": 50.0, "newbalanceDest": 50.01},
        ]
    )
    bundle = joblib.load(PICKLE_PATH)
    reference = bundle["model"].predict_proba(engineer_features(edge)[FEATURE_COLS])[:, 1]
    ported = [score.score_transaction(r)["fraud_probability"] for r in edge.to_dict("records")]

    for got, want in zip(ported, reference):
        assert got == pytest.approx(want, abs=MAX_ABS_DIFF)
