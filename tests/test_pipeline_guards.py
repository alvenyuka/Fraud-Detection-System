"""Regression tests for the two leaks this repository fixed, and for the
calibration split and the scoring entry point that every result depends on.

1. Tuning rows must never be reporting rows. Hyperparameters chosen to maximise
   PR-AUC on a window make any metric later reported on that window optimistic.
2. The decision threshold must be chosen on calibration labels, never on test
   labels. Choosing it on the test rows makes precision and recall
   oracle-thresholded.

Both leaks were present once and were fixed. Without a test either can come
back through an innocent-looking edit, and no metric would show it.
"""
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

import validate
from config import FOLDS, HOLDOUT_STEP_SHARE, SPLIT_STEP, TUNING_SPLITS
from features import split_off_latest_steps
from predict import load_model, score_dataframe

SMALL_XGB_PARAMS = {
    "n_estimators": 10,
    "max_depth": 2,
    "learning_rate": 0.3,
    "random_state": 0,
    "n_jobs": 1,
    "verbosity": 0,
    "early_stopping_rounds": 30,  # present in tuned params; run_one_fold must drop it
}


def _synthetic_paysim(n_steps=100, rows_per_step=40, seed=0):
    """Balance fields where fraud drains the sender, plus a step column."""
    rng = np.random.default_rng(seed)
    n = n_steps * rows_per_step
    step = np.repeat(np.arange(1, n_steps + 1), rows_per_step)
    fraud = rng.random(n) < 0.08
    old_orig = rng.lognormal(8, 1, n)
    amount = np.where(fraud, old_orig, old_orig * rng.random(n) * 0.5)
    old_dest = rng.lognormal(7, 1, n)
    frame = pd.DataFrame(
        {
            "step": step,
            "amount": amount,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": np.where(fraud, 0.0, old_orig - amount),
            "oldbalanceDest": old_dest,
            "newbalanceDest": np.where(fraud, old_dest, old_dest + amount),
            "isFraud": fraud.astype(int),
        }
    )
    from features import engineer_features

    return engineer_features(frame)


# --------------------------------------------------------------------------
# Leak 1: tuning windows against reporting windows
# --------------------------------------------------------------------------

def test_every_tuning_window_ends_before_every_reporting_window():
    last_tuning_step = max(end for _, end in TUNING_SPLITS)
    first_fold_train_end = min(start for start, _ in FOLDS)
    assert last_tuning_step <= first_fold_train_end
    assert last_tuning_step < SPLIT_STEP


def test_folds_are_ordered_and_do_not_overlap():
    for (start, end), (next_start, _) in zip(FOLDS, FOLDS[1:]):
        assert start < end <= next_start


# --------------------------------------------------------------------------
# Leak 2: the threshold is chosen on calibration labels
# --------------------------------------------------------------------------

def test_run_one_fold_picks_its_threshold_on_the_calibration_split(monkeypatch):
    df = _synthetic_paysim()
    train_end, test_end = 80, 100
    _, expected_calibration = split_off_latest_steps(df[df["step"] <= train_end])
    n_test = int(((df["step"] > train_end) & (df["step"] <= test_end)).sum())

    seen = {}

    def recording_threshold(y_true, predicted_probs):
        seen["n"] = len(y_true)
        seen["fraud"] = int(np.sum(y_true))
        return 0.5

    monkeypatch.setattr(validate, "pick_best_threshold", recording_threshold)
    result = validate.run_one_fold(df, train_end, test_end, SMALL_XGB_PARAMS)

    assert seen["n"] == len(expected_calibration)
    assert seen["fraud"] == int(expected_calibration["isFraud"].sum())
    assert seen["n"] != n_test
    assert result["n_test"] == n_test


# --------------------------------------------------------------------------
# The time-ordered calibration split
# --------------------------------------------------------------------------

def test_calibration_split_is_the_latest_steps():
    df = _synthetic_paysim(n_steps=490)
    fit, calibration = split_off_latest_steps(df)
    assert fit["step"].max() < calibration["step"].min()
    assert len(fit) + len(calibration) == len(df)
    n_calibration_steps = calibration["step"].nunique()
    assert n_calibration_steps == int(490 * HOLDOUT_STEP_SHARE)
    assert calibration["step"].max() == 490


def test_calibration_split_refuses_a_side_without_fraud():
    df = _synthetic_paysim()
    df.loc[df["step"] > 80, "isFraud"] = 0
    with pytest.raises(ValueError):
        split_off_latest_steps(df)


# --------------------------------------------------------------------------
# predict.score_dataframe and predict.load_model
# --------------------------------------------------------------------------

class _RecordingModel:
    """Stands in for the calibrated model: returns the drain ratio as the score."""

    def __init__(self):
        self.columns_seen = None

    def predict_proba(self, X):
        self.columns_seen = list(X.columns)
        p = np.clip(X["orig_drain_ratio"].to_numpy(), 0, 1)
        return np.column_stack([1 - p, p])


def _transactions(with_type=True):
    frame = pd.DataFrame(
        {
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT"],
            "amount": [100.0, 100.0, 10.0],
            "oldbalanceOrg": [100.0, 100.0, 100.0],
            "newbalanceOrig": [0.0, 0.0, 90.0],
            "oldbalanceDest": [0.0, 0.0, 0.0],
            "newbalanceDest": [100.0, 100.0, 10.0],
        }
    )
    return frame if with_type else frame.drop(columns="type")


def _artifact(feature_cols=None):
    artifact = {"model": _RecordingModel(), "operating_threshold": 0.5}
    if feature_cols is not None:
        artifact["feature_cols"] = feature_cols
    return artifact


def test_only_fraud_active_types_are_scored():
    scored = score_dataframe(_transactions(), _artifact())
    assert np.isnan(scored.loc[1, "fraud_score"])
    assert scored.loc[1, "fraud_flag"] == 0
    assert scored.loc[0, "fraud_flag"] == 1
    assert scored.loc[2, "fraud_flag"] == 0


def test_missing_input_columns_raise_a_clear_error():
    with pytest.raises(ValueError, match="newbalanceDest"):
        score_dataframe(_transactions().drop(columns="newbalanceDest"), _artifact())


def test_a_file_without_type_is_rejected_unless_assumed_active():
    with pytest.raises(ValueError, match="type"):
        score_dataframe(_transactions(with_type=False), _artifact())
    scored = score_dataframe(_transactions(with_type=False), _artifact(), assume_active=True)
    assert scored["fraud_score"].notna().all()


def test_scoring_uses_the_feature_list_saved_with_the_model():
    saved = ["orig_drain_ratio", "amount"]
    artifact = _artifact(feature_cols=saved)
    score_dataframe(_transactions(), artifact)
    assert artifact["model"].columns_seen == saved


def test_load_model_warns_when_library_versions_differ(tmp_path):
    path = tmp_path / "model.pkl"
    joblib.dump({"model": "stand-in", "operating_threshold": 0.5,
                 "versions": {"scikit-learn": "0.0.1", "xgboost": "0.0.1"}}, path)
    with pytest.warns(UserWarning, match="scikit-learn"):
        load_model(str(path))
    with pytest.raises(RuntimeError):
        load_model(str(path), strict=True)


def test_load_model_is_silent_when_versions_match(tmp_path):
    import sklearn
    import xgboost

    path = tmp_path / "model.pkl"
    joblib.dump({"model": "stand-in", "operating_threshold": 0.5,
                 "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}}, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_model(str(path))


# --------------------------------------------------------------------------
# Tuning records the tree count it actually scored (needs optuna, which CI skips)
# --------------------------------------------------------------------------

def test_a_trial_records_its_median_early_stopped_tree_count():
    optuna = pytest.importorskip("optuna")
    import tune

    df = _synthetic_paysim(n_steps=350, rows_per_step=12, seed=1)
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = {**SMALL_XGB_PARAMS, "n_estimators": 200, "early_stopping_rounds": 5, "eval_metric": "aucpr"}
    tune.score_one_trial(params, df, trial)

    counts = trial.user_attrs["tree_counts"]
    assert len(counts) == len(TUNING_SPLITS)
    assert all(1 <= c <= 200 for c in counts)
    assert trial.user_attrs["n_estimators"] == int(np.median(counts))
