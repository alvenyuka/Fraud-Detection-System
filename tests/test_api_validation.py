"""Input validation in api/score.py, the function behind the live demo.

XGBoost treats NaN as missing and follows each split's default branch, while
the pure-Python traversal compares NaN as "not less than" and goes right. The
two disagree on exactly the inputs the parity test never generates, so the API
must refuse them before scoring. The same goes for infinities, negative
amounts or balances, and oversized request bodies.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_PATH = REPO_ROOT / "model" / "model_export.json"

# api/score.py fetches the export over HTTP at import time unless this is set.
os.environ["FRAUD_MODEL_EXPORT_PATH"] = str(EXPORT_PATH)

pytestmark = pytest.mark.skipif(not EXPORT_PATH.exists(), reason="needs the committed model export")


@pytest.fixture(scope="module")
def score():
    sys.path.insert(0, str(REPO_ROOT / "api"))
    import score as module  # noqa: PLC0415 -- imported after the env var is set

    return module


VALID = {
    "amount": 500.0,
    "oldbalanceOrg": 1000.0,
    "newbalanceOrig": 500.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 500.0,
}


def test_a_valid_body_is_accepted(score):
    assert score.parse_transaction(dict(VALID)) == VALID


def test_nan_is_rejected(score):
    # json.loads accepts the bare NaN literal, which is how one would arrive.
    body = json.loads('{"amount": NaN, "oldbalanceOrg": 1, "newbalanceOrig": 0,'
                      ' "oldbalanceDest": 0, "newbalanceDest": 1}')
    with pytest.raises(ValueError, match="finite"):
        score.parse_transaction(body)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), -1.0, "Infinity"])
def test_infinite_and_negative_values_are_rejected(score, bad):
    with pytest.raises(ValueError):
        score.parse_transaction({**VALID, "oldbalanceOrg": bad})


@pytest.mark.parametrize("bad", ["abc", None, True, [1]])
def test_non_numeric_values_are_rejected(score, bad):
    with pytest.raises(ValueError, match="number"):
        score.parse_transaction({**VALID, "amount": bad})


def test_missing_fields_are_named(score):
    body = dict(VALID)
    del body["newbalanceDest"]
    with pytest.raises(ValueError, match="newbalanceDest"):
        score.parse_transaction(body)


def test_oversized_bodies_are_rejected(score):
    score.check_body_length(score.MAX_BODY_BYTES)
    with pytest.raises(ValueError, match="too large"):
        score.check_body_length(score.MAX_BODY_BYTES + 1)
