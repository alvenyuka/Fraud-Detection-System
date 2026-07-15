"""
export_model_json.py — exports the shipped model to a dependency-free JSON
format for the Vercel-hosted interactive demo.

xgboost + scikit-learn + scipy together are ~245MB installed, which is at or
over Vercel's serverless function size limit. Rather than bundle those
packages, this exports the trained trees (native XGBoost JSON format, full
float32 precision) and the fitted isotonic calibration breakpoints as plain
data. The Vercel function re-implements tree traversal and calibration in
pure Python (stdlib only) -- see api/score.py.

Validated to match the real model's predict_proba to within float32
precision (max abs diff 0.0000000994 across 5000 real held-out rows) --
see the validation run in this project's session notes before trusting
this file's output.

Usage
-----
    python src/export_model_json.py
"""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "model" / "xgb_fraud_model.pkl"
OUT_PATH = ROOT / "model" / "model_export.json"
TMP_XGB_JSON = ROOT / "model" / "_tmp_xgb_export.json"


def export() -> None:
    artifact = joblib.load(MODEL_PATH)
    calibrated = artifact["model"]
    cc = calibrated.calibrated_classifiers_[0]
    raw_xgb = cc.estimator.estimator  # unwrap FrozenEstimator
    booster = raw_xgb.get_booster()

    # Native format keeps full float32 precision (unlike get_dump's text format).
    booster.save_model(str(TMP_XGB_JSON))
    xgb_model = json.loads(TMP_XGB_JSON.read_text())
    learner = xgb_model["learner"]

    base_score = float(learner["learner_model_param"]["base_score"].strip("[]"))
    feature_names = learner["feature_names"]
    trees = learner["gradient_booster"]["model"]["trees"]

    # Keep only the fields the pure-Python traversal actually needs.
    compact_trees = [
        {
            "left": t["left_children"],
            "right": t["right_children"],
            "feat": t["split_indices"],
            "cond": t["split_conditions"],
            "default_left": t["default_left"],
        }
        for t in trees
    ]

    iso = cc.calibrators[0]

    export_data = {
        "feature_names": feature_names,
        "base_score": base_score,
        "trees": compact_trees,
        "isotonic_x": iso.X_thresholds_.tolist(),
        "isotonic_y": iso.y_thresholds_.tolist(),
        "operating_threshold": artifact["operating_threshold"],
    }

    OUT_PATH.write_text(json.dumps(export_data))
    TMP_XGB_JSON.unlink()

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Exported {len(compact_trees)} trees -> {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    export()
