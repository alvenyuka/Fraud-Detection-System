"""
validate.py — Step 3 of the model build-up: walk-forward validation.

The first version of this project only ever tested the model on one
train/test split (train on steps 1-490, test on 491-743). That tells us the
model worked once, on one slice of time — it doesn't tell us whether that
was a lucky split or a model that actually holds up.

This script repeats the same train -> calibrate -> test recipe from
train.py four times, each time moving the split point further forward in
time ("walk-forward"). If the scores stay similar across all four folds, that's
real evidence the model is stable over time, not a one-off result.

    Fold 1: train on steps  1-350, test on 351-450
    Fold 2: train on steps  1-450, test on 451-550
    Fold 3: train on steps  1-550, test on 551-650
    Fold 4: train on steps  1-650, test on 651-743

It also saves the small chart-ready files the dashboard reads:
dashboard/data/walk_forward_results.csv, pr_curve.csv, calibration_curve.csv,
confusion_matrix.json, metrics_summary.json.

Usage
-----
    python src/validate.py --data PS_20174392719_1491204439457_log.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, engineer_features, load_and_filter, pick_best_threshold  # noqa: E402
from train import load_xgb_params  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FOLDS = [
    (350, 450),
    (450, 550),
    (550, 650),
    (650, 743),
]

DASHBOARD_DATA_DIR = Path(__file__).parent.parent / "dashboard" / "data"


def run_one_fold(df: pd.DataFrame, train_end_step: int, test_end_step: int, xgb_params: dict) -> dict:
    """Train on everything up to train_end_step, test on the steps right after it."""
    train_period = df[df["step"] <= train_end_step]
    test_period = df[(df["step"] > train_end_step) & (df["step"] <= test_end_step)]

    # Same calibration approach as train.py: hold out part of the training
    # data so the probability calibration step isn't graded on data the
    # model already memorised.
    fit_rows, calibration_rows = train_test_split(
        train_period, test_size=0.2, stratify=train_period["isFraud"], random_state=42
    )

    X_fit, y_fit = fit_rows[FEATURE_COLS], fit_rows["isFraud"]
    X_calibration, y_calibration = calibration_rows[FEATURE_COLS], calibration_rows["isFraud"]
    X_test, y_test = test_period[FEATURE_COLS], test_period["isFraud"]

    # early_stopping_rounds only makes sense during tuning (it needs a
    # validation set); drop it here so a plain .fit() call works.
    clean_params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    base_model = XGBClassifier(**clean_params)
    base_model.fit(X_fit, y_fit)

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="isotonic")
    calibrated_model.fit(X_calibration, y_calibration)

    predicted_probs = calibrated_model.predict_proba(X_test)[:, 1]
    threshold = pick_best_threshold(y_test.to_numpy(), predicted_probs)
    predicted_fraud = (predicted_probs >= threshold).astype(int)

    return {
        "train_end_step": train_end_step,
        "test_end_step": test_end_step,
        "n_train": len(train_period),
        "n_test": len(test_period),
        "fraud_rate_test_pct": round(100 * y_test.mean(), 4),
        "threshold": round(threshold, 4),
        "PR-AUC": round(average_precision_score(y_test, predicted_probs), 4),
        "ROC-AUC": round(roc_auc_score(y_test, predicted_probs), 4),
        "Precision": round(precision_score(y_test, predicted_fraud), 4),
        "Recall": round(recall_score(y_test, predicted_fraud), 4),
        "F1": round(f1_score(y_test, predicted_fraud), 4),
        "Brier": round(brier_score_loss(y_test, predicted_probs), 6),
        # kept only for building charts below, not part of the saved summary table
        "_y_test": y_test.to_numpy(),
        "_predicted_probs": predicted_probs,
        "_predicted_fraud": predicted_fraud,
    }


def validate(data_path: str) -> None:
    xgb_params = load_xgb_params()
    df = load_and_filter(data_path)
    df = engineer_features(df)

    log.info("Running %d walk-forward folds ...", len(FOLDS))
    fold_results = []
    for train_end_step, test_end_step in FOLDS:
        log.info("Fold: train on steps <= %d, test on %d-%d", train_end_step, train_end_step + 1, test_end_step)
        result = run_one_fold(df, train_end_step, test_end_step, xgb_params)
        fold_results.append(result)
        log.info(
            "  PR-AUC=%.4f  ROC-AUC=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f  (threshold=%.4f)",
            result["PR-AUC"], result["ROC-AUC"], result["Precision"],
            result["Recall"], result["F1"], result["threshold"],
        )

    # Average each metric across all 4 folds, plus how much it varies (std) —
    # a small std means the model behaves consistently across time.
    metric_names = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Brier"]
    summary = {
        "n_folds": len(fold_results),
        "folds": [{k: v for k, v in r.items() if not k.startswith("_")} for r in fold_results],
    }
    for metric in metric_names:
        values = [r[metric] for r in fold_results]
        summary[metric] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}

    log.info("Walk-forward summary (mean +/- std across %d folds):", len(fold_results))
    for metric in metric_names:
        log.info("  %-10s %.4f +/- %.4f", metric, summary[metric]["mean"], summary[metric]["std"])

    save_dashboard_artifacts(fold_results, summary)


def save_dashboard_artifacts(fold_results: list, summary: dict) -> None:
    """Write the small files dashboard/app.py's Model Performance tab reads."""
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fold_table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in fold_results])
    fold_table.to_csv(DASHBOARD_DATA_DIR / "walk_forward_results.csv", index=False)

    # Charts (PR curve, calibration curve, confusion matrix) come from the
    # most recent fold, since it's trained on the most data — the closest
    # thing we have to "the model as it would ship today".
    most_recent_fold = fold_results[-1]

    precision_vals, recall_vals, _ = precision_recall_curve(
        most_recent_fold["_y_test"], most_recent_fold["_predicted_probs"]
    )
    pd.DataFrame({"precision": precision_vals, "recall": recall_vals}).to_csv(
        DASHBOARD_DATA_DIR / "pr_curve.csv", index=False
    )

    mean_predicted, fraction_actually_fraud = calibration_curve(
        most_recent_fold["_y_test"], most_recent_fold["_predicted_probs"], n_bins=10, strategy="quantile"
    )
    pd.DataFrame({"mean_predicted": mean_predicted, "fraction_positive": fraction_actually_fraud}).to_csv(
        DASHBOARD_DATA_DIR / "calibration_curve.csv", index=False
    )

    matrix = confusion_matrix(most_recent_fold["_y_test"], most_recent_fold["_predicted_fraud"]).tolist()
    (DASHBOARD_DATA_DIR / "confusion_matrix.json").write_text(
        json.dumps({"labels": ["Legitimate", "Fraud"], "matrix": matrix}, indent=2)
    )

    (DASHBOARD_DATA_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Dashboard artifacts written -> %s", DASHBOARD_DATA_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation for the fraud classifier.")
    p.add_argument("--data", required=True, metavar="CSV", help="Path to PaySim CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(args.data)
