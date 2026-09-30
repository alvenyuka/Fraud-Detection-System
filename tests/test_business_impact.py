"""The arithmetic behind the README's business-impact table."""
import numpy as np
import pytest

from business_impact import screening_outcome


def test_value_is_split_between_stopped_and_missed():
    flagged = np.array([1, 0, 1, 0], bool)
    fraud = np.array([1, 1, 0, 0], bool)
    amount = np.array([500.0, 300.0, 50.0, 20.0])
    r = screening_outcome(flagged, fraud, amount)
    assert r["fraud_value_stopped"] == 500
    assert r["fraud_value_missed"] == 300
    assert r["share_of_fraud_value_stopped"] == pytest.approx(500 / 800)
    assert (r["frauds_caught"], r["frauds_missed"]) == (1, 1)


def test_false_alarms_count_honest_transactions_only():
    flagged = np.array([1, 1, 1], bool)
    fraud = np.array([1, 0, 0], bool)
    amount = np.array([10.0, 20.0, 30.0])
    r = screening_outcome(flagged, fraud, amount)
    assert r["honest_transactions_frozen"] == 2
    assert r["honest_value_frozen"] == 50


def test_no_screening_stops_nothing():
    fraud = np.array([1, 0, 1], bool)
    r = screening_outcome(np.zeros(3, bool), fraud, np.array([1.0, 2.0, 3.0]))
    assert r["fraud_value_stopped"] == 0
    assert r["share_of_fraud_value_stopped"] == 0
    assert r["honest_transactions_frozen"] == 0
