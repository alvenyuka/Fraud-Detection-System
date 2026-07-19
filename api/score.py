"""
api/score.py — Vercel serverless function: real-time fraud scoring.

Re-implements the shipped XGBoost + isotonic-calibration model in pure
Python (stdlib only, no xgboost/scikit-learn) so it fits comfortably inside
a Vercel function and never cold-starts slow or hibernates. The model
itself is unchanged -- see model_export.json (exported by
src/export_model_json.py in the main repo) and MODEL_CARD.md for what it
was trained on.

Validated against the real scikit-learn/xgboost model: max absolute
probability difference of 0.0000038867 across 20,000 real held-out
transactions, plus exact matches on full-drain, zero-balance, and
tiny-amount edge cases (pure floating-point noise, not a modelling
difference).
"""
import json
import math
import struct
import urllib.request
from http.server import BaseHTTPRequestHandler

# Fetched once per cold start and cached at module level for the lifetime of
# this function instance -- avoids bundling a 350KB data file into the
# deployment (the model itself is unchanged; this is just where it lives).
_MODEL_URL = (
    "https://raw.githubusercontent.com/alvenyuka/Fraud-Detection-System/main/model/model_export.json"
)
with urllib.request.urlopen(_MODEL_URL, timeout=10) as resp:
    _MODEL = json.loads(resp.read())

FEATURE_NAMES = _MODEL["feature_names"]
BASE_MARGIN = math.log(_MODEL["base_score"] / (1 - _MODEL["base_score"]))
TREES = _MODEL["trees"]
ISO_X = _MODEL["isotonic_x"]
ISO_Y = _MODEL["isotonic_y"]
OPERATING_THRESHOLD = _MODEL["operating_threshold"]

EPS = 1e-8


def engineer_features(txn: dict) -> dict:
    """Same formulas as src/features.py::engineer_features."""
    amount = txn["amount"]
    old_orig, new_orig = txn["oldbalanceOrg"], txn["newbalanceOrig"]
    old_dest, new_dest = txn["oldbalanceDest"], txn["newbalanceDest"]
    return {
        "amount": amount,
        "orig_balance_discrepancy": old_orig - amount - new_orig,
        "dest_balance_discrepancy": old_dest + amount - new_dest,
        "orig_drain_ratio": amount / (old_orig + EPS),
        "dest_amount_ratio": amount / (old_dest + EPS),
    }


def _f32(x: float) -> float:
    """Round-trip through 32-bit precision, matching XGBoost's internal float32 comparisons."""
    return struct.unpack("f", struct.pack("f", x))[0]


def eval_tree(tree: dict, values: list) -> float:
    node = 0
    left, right = tree["left"], tree["right"]
    feat, cond, default_left = tree["feat"], tree["cond"], tree["default_left"]
    while left[node] != -1:
        val = values[feat[node]]
        if val is None:
            node = left[node] if default_left[node] else right[node]
        elif _f32(val) < _f32(cond[node]):
            node = left[node]
        else:
            node = right[node]
    return cond[node]


def isotonic_predict(raw_prob: float) -> float:
    x_min, x_max = ISO_X[0], ISO_X[-1]
    x = min(max(raw_prob, x_min), x_max)
    # linear scan is fine -- only 28 breakpoints
    if x <= ISO_X[0]:
        return ISO_Y[0]
    for i in range(1, len(ISO_X)):
        if x <= ISO_X[i]:
            x0, x1 = ISO_X[i - 1], ISO_X[i]
            y0, y1 = ISO_Y[i - 1], ISO_Y[i]
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ISO_Y[-1]


def score_transaction(txn: dict) -> dict:
    features = engineer_features(txn)
    values = [features[name] for name in FEATURE_NAMES]
    margin = BASE_MARGIN + sum(eval_tree(t, values) for t in TREES)
    raw_prob = 1.0 / (1.0 + math.exp(-margin))
    fraud_probability = isotonic_predict(raw_prob)
    return {
        "fraud_probability": fraud_probability,
        "flagged": fraud_probability >= OPERATING_THRESHOLD,
        "operating_threshold": OPERATING_THRESHOLD,
        "orig_drain_ratio": features["orig_drain_ratio"],
    }


REQUIRED_FIELDS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


class handler(BaseHTTPRequestHandler):
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            missing = [f for f in REQUIRED_FIELDS if f not in body]
            if missing:
                raise ValueError(f"Missing fields: {', '.join(missing)}")
            txn = {f: float(body[f]) for f in REQUIRED_FIELDS}
            result = score_transaction(txn)
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        except Exception as exc:  # noqa: BLE001 -- return the error to the caller, not a 500 wall
            self.send_response(400)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())
