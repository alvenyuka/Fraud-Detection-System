"""
tune.py — Step 2 of the model build-up: hyperparameter tuning.

The original model (src/train.py) used one fixed set of XGBoost settings that
were never actually tested against alternatives. This script tries a range of
settings and keeps the combination that scores best, so the final model isn't
just "whatever numbers we guessed the first time."

How we test each combination of settings ("trial"):
  1. Take only the training period (steps 1-650). The last part of the
     dataset (steps 651-743) is never touched here — it's saved for the
     walk-forward check in src/validate.py and for train.py's own test set.
  2. Inside that training period, split time into 3 smaller before/after
     windows (see TUNING_SPLITS below). For each window, train on the
     "before" part and score on the "after" part.
  3. Average the score (PR-AUC — the right metric for rare-event problems
     like fraud) across the 3 windows. That average is the trial's score.
Optuna (a hyperparameter search library) tries ~40 different combinations
and remembers which one scored highest.

Usage
-----
    python src/tune.py --data PS_20174392719_1491204439457_log.csv --trials 40
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, engineer_features, load_and_filter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep Optuna's own logs quiet, we log ourselves below

# Three small "before -> after" windows used only for tuning. All of them sit
# inside the training period (step <= 650), so tuning never sees the data
# that validate.py or train.py use to report final results.
TUNING_SPLITS = [
    (250, 350),  # train on steps <= 250, score on steps 251-350
    (350, 450),
    (450, 550),
]

BEST_PARAMS_PATH = Path(__file__).parent.parent / "model" / "best_params.json"


def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    """
    One trial = one guess at a full set of XGBoost settings.

    Grouped by what each setting controls, so it's clear why each one is here:
      - Tree shape:        max_depth, min_child_weight
      - Learning behaviour: learning_rate, n_estimators (capped, early stopping decides the real count)
      - Randomness/overfitting control: subsample, colsample_bytree
      - Regularisation (penalise overly complex trees): gamma, reg_alpha, reg_lambda
      - Class imbalance (fraud is rare): scale_pos_weight
    """
    return {
        "n_estimators": 500,
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2, 30, log=True),
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 30,
    }


def score_one_trial(params: dict, df) -> float:
    """Train + score this parameter set on each of the 3 tuning windows, return the average."""
    window_scores = []
    for train_end_step, window_end_step in TUNING_SPLITS:
        train_window = df[df["step"] <= train_end_step]
        test_window = df[(df["step"] > train_end_step) & (df["step"] <= window_end_step)]

        if test_window["isFraud"].sum() == 0:
            continue  # skip a window with no fraud cases — nothing to score

        X_train, y_train = train_window[FEATURE_COLS], train_window["isFraud"]
        X_test, y_test = test_window[FEATURE_COLS], test_window["isFraud"]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predicted_probs = model.predict_proba(X_test)[:, 1]
        window_scores.append(average_precision_score(y_test, predicted_probs))

    return float(np.mean(window_scores)) if window_scores else 0.0


def tune(data_path: str, n_trials: int, out_path: str) -> None:
    df = load_and_filter(data_path)
    df = engineer_features(df)

    # Only steps <= 650 are used for tuning — keeps the final held-out period untouched.
    training_period = df[df["step"] <= 650]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    log.info("Starting hyperparameter search: %d trials over %d rows ...", n_trials, len(training_period))
    study.optimize(
        lambda trial: score_one_trial(suggest_hyperparameters(trial), training_period),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    log.info("Best average PR-AUC across the 3 tuning windows: %.4f", study.best_value)
    log.info("Best settings found: %s", study.best_params)

    best_params = dict(study.best_params)
    best_params["n_estimators"] = 500  # the final training run doesn't use early stopping, so fix this

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best_params, indent=2))
    log.info("Saved tuned settings -> %s (train.py will pick these up automatically)", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune XGBoost hyperparameters for the fraud classifier.")
    p.add_argument("--data", required=True, metavar="CSV", help="Path to PaySim CSV.")
    p.add_argument("--trials", type=int, default=40, help="Number of hyperparameter combinations to try (default: 40).")
    p.add_argument(
        "--out",
        default=str(BEST_PARAMS_PATH),
        metavar="JSON",
        help="Where to save the best settings found.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune(args.data, args.trials, args.out)
