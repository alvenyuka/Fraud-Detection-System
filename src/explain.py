"""
explain.py — Step 7 of the model build-up: richer dashboard diagnostics.

Loads the already-shipped model (no retraining here) and scores the same
held-out test period train.py evaluates on, then saves three small files the
dashboard's Model Performance tab reads:

  - feature_importance.json    — mean |SHAP| per feature, as a percentage
  - probability_distribution.csv — binned histogram of predicted probabilities,
                                    fraud vs. legitimate (binned, not one row
                                    per transaction, so the file stays small
                                    enough to commit like every other
                                    dashboard/data file)
  - threshold_cost_curve.csv   — precision/recall/cost at a range of
                                  candidate decision thresholds, so the
                                  dashboard's threshold slider has real
                                  numbers to look up instead of recomputing
                                  live

Usage
-----
    python src/explain.py --data PS_20174392719_1491204439457_log.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, engineer_features, load_and_filter  # noqa: E402
from train import SPLIT_STEP, time_based_split  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "model" / "xgb_fraud_model.pkl"
DASHBOARD_DATA_DIR = Path(__file__).parent.parent / "dashboard" / "data"

N_PROBABILITY_BINS = 50
COST_PER_MISSED_FRAUD = 1000
COST_PER_FALSE_ALARM = 10


def get_raw_xgboost_model(artifact: dict):
    """Same unwrapping dashboard/app.py does — SHAP needs the tree model, not the calibration wrapper."""
    calibrated_model = artifact["model"]
    first_fold = calibrated_model.calibrated_classifiers_[0]
    return first_fold.estimator.estimator


def save_feature_importance(raw_xgb_model, X_sample: pd.DataFrame) -> None:
    """Mean |SHAP| per feature, as a percentage — answers 'does one feature dominate?'"""
    explainer = shap.TreeExplainer(raw_xgb_model)
    shap_values = explainer(X_sample)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    percentages = 100 * mean_abs_shap / mean_abs_shap.sum()

    importance = sorted(
        [{"feature": f, "importance_pct": round(float(p), 2)} for f, p in zip(FEATURE_COLS, percentages)],
        key=lambda row: -row["importance_pct"],
    )
    out_path = DASHBOARD_DATA_DIR / "feature_importance.json"
    out_path.write_text(json.dumps(importance, indent=2))
    log.info("Feature importance saved -> %s", out_path)
    for row in importance:
        log.info("  %-28s %5.2f%%", row["feature"], row["importance_pct"])


def save_probability_distribution(probs: np.ndarray, y_true: np.ndarray) -> None:
    """Binned histogram of predicted probabilities, fraud vs. legitimate."""
    bin_edges = np.linspace(0, 1, N_PROBABILITY_BINS + 1)
    fraud_counts, _ = np.histogram(probs[y_true == 1], bins=bin_edges)
    legit_counts, _ = np.histogram(probs[y_true == 0], bins=bin_edges)

    result = pd.DataFrame({
        "bin_start": bin_edges[:-1],
        "bin_end": bin_edges[1:],
        "count_legitimate": legit_counts,
        "count_fraud": fraud_counts,
    })
    out_path = DASHBOARD_DATA_DIR / "probability_distribution.csv"
    result.to_csv(out_path, index=False)
    log.info("Probability distribution saved -> %s", out_path)


def save_threshold_cost_curve(probs: np.ndarray, y_true: np.ndarray) -> None:
    """Precision/recall/cost at a spread of candidate thresholds, for the dashboard's slider."""
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        missed_fraud = int(((preds == 0) & (y_true == 1)).sum())
        false_alarms = int(((preds == 1) & (y_true == 0)).sum())
        cost = COST_PER_MISSED_FRAUD * missed_fraud + COST_PER_FALSE_ALARM * false_alarms
        rows.append({
            "threshold": round(float(threshold), 4),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "cost": cost,
        })
    result = pd.DataFrame(rows)
    out_path = DASHBOARD_DATA_DIR / "threshold_cost_curve.csv"
    result.to_csv(out_path, index=False)
    log.info("Threshold/cost curve saved -> %s", out_path)


def explain(data_path: str) -> None:
    artifact = joblib.load(MODEL_PATH)
    raw_xgb_model = get_raw_xgboost_model(artifact)

    df = load_and_filter(data_path)
    df = engineer_features(df)
    _, test_df = time_based_split(df, SPLIT_STEP)

    X_test, y_test = test_df[FEATURE_COLS], test_df["isFraud"].to_numpy()
    log.info("Scoring %d held-out test rows with the shipped model ...", len(X_test))
    probs = artifact["model"].predict_proba(X_test)[:, 1]

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # SHAP on a 2,000-row sample -- consistent with the sample size already
    # used and documented elsewhere in this project (MODEL_CARD.md).
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    save_feature_importance(raw_xgb_model, sample)

    save_probability_distribution(probs, y_test)
    save_threshold_cost_curve(probs, y_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dashboard diagnostic artifacts from the shipped model.")
    p.add_argument("--data", required=True, metavar="CSV", help="Path to PaySim CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explain(args.data)
