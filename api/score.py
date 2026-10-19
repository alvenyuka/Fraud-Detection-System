"""
api/score.py: Vercel serverless function for real-time fraud scoring.

Re-implements the shipped XGBoost + isotonic-calibration model in pure
Python (stdlib only, no xgboost/scikit-learn) so it fits comfortably inside
a Vercel function and never cold-starts slow or hibernates. The model
itself is unchanged -- see model_export.json (exported by
src/export_model_json.py in the main repo) and MODEL_CARD.md for what it
was trained on.

Parity with the real model is enforced by tests/test_export_parity.py rather
than asserted here. That test scores a fixed-seed sample of 20,000
transactions through both this file's scoring path and the committed
scikit-learn/xgboost pickle. For the model trained on 2026-10-02 the measured
worst case is about 6.6e-08 in
absolute probability, concentrated on rows that land exactly on a tree split
threshold or an isotonic breakpoint; the mean disagreement is around 1.4e-10.

Requests are validated before scoring: the five fields must be present, finite
and non-negative, and the body is capped at MAX_BODY_BYTES. XGBoost treats NaN
as missing while the traversal below would send it right, so parity only holds
for finite inputs, and the validation keeps the port inside that range.
"""
import json
import math
import os
import struct
import urllib.request
from http.server import BaseHTTPRequestHandler

# Fetched once per cold start and cached at module level for the lifetime of
# this function instance, which avoids bundling the export into the deployment.
#
# This URL follows the main branch on purpose: pushing a new
# model/model_export.json to main changes the live demo's scores at the next
# cold start, with no separate deploy. Retrain, re-export, run
# tests/test_export_parity.py and update the docs before pushing an export.
_MODEL_URL = (
    "https://raw.githubusercontent.com/alvenyuka/Fraud-Detection-System/main/model/model_export.json"
)

# The parity test needs to import this module without a network call, and CI
# has no reason to reach GitHub raw to run it. Setting this variable points the
# loader at a local model_export.json instead. It is unset in the deployment,
# which keeps the serverless path exactly as it was.
_LOCAL_EXPORT = os.environ.get("FRAUD_MODEL_EXPORT_PATH")

UNAVAILABLE_MESSAGE = "Model temporarily unavailable, please retry"


class ModelUnavailable(RuntimeError):
    """The model export could not be fetched or read, so nothing can be scored."""


# Filled by load_model() on the first successful load and reused afterwards. A
# failed load is not cached, so the next request tries again.
_MODEL_STATE = None


def _read_export() -> dict:
    if _LOCAL_EXPORT:
        with open(_LOCAL_EXPORT, encoding="utf-8") as fh:
            return json.load(fh)
    with urllib.request.urlopen(_MODEL_URL, timeout=10) as resp:
        return json.loads(resp.read())


def load_model() -> dict:
    """Return the parsed model, fetching it on first use; raise ModelUnavailable on any failure."""
    global _MODEL_STATE
    if _MODEL_STATE is None:
        try:
            model = _read_export()
            state = {
                "feature_names": model["feature_names"],
                "base_margin": math.log(model["base_score"] / (1 - model["base_score"])),
                "trees": model["trees"],
                "iso_x": model["isotonic_x"],
                "iso_y": model["isotonic_y"],
                "operating_threshold": model["operating_threshold"],
            }
        except Exception as exc:  # noqa: BLE001 -- network, HTTP, JSON or shape: all mean "no model"
            raise ModelUnavailable(UNAVAILABLE_MESSAGE) from exc
        _MODEL_STATE = state
    return _MODEL_STATE


# Load at cold start as before, so the first request is not slower; if GitHub
# raw is unreachable now, the first request retries instead of the import failing.
try:
    load_model()
except ModelUnavailable:
    pass

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
    """Round a Python float (64-bit) to the nearest 32-bit float.

    XGBoost stores feature values and split points as 32-bit floats and compares them
    at that precision. A value just below a split point in 64-bit can be equal to it
    in 32-bit, and it would then go down the other branch. Rounding both sides here
    makes this file choose the same branch as XGBoost.
    """
    packed = struct.pack("f", x)  # "f" is a C float, 4 bytes
    return struct.unpack("f", packed)[0]


def eval_tree(tree: dict, values: list) -> float:
    """Walk one exported XGBoost tree for one transaction and return the leaf value.

    The tree is stored as parallel lists: entry i of each list describes node i,
    and node 0 is the root. A left child of -1 marks a leaf. A leaf keeps its
    output in the same list that holds split points for other nodes ("cond").
    """
    left_child = tree["left"]
    right_child = tree["right"]
    feature_index = tree["feat"]
    split_value = tree["cond"]
    default_left = tree["default_left"]

    node = 0
    while left_child[node] != -1:
        value = values[feature_index[node]]

        if value is None:
            # A missing value follows the branch XGBoost chose during training.
            if default_left[node]:
                node = left_child[node]
            else:
                node = right_child[node]
        elif _f32(value) < _f32(split_value[node]):
            node = left_child[node]
        else:
            node = right_child[node]

    return split_value[node]


def isotonic_predict(raw_prob: float) -> float:
    """Map a raw probability through the exported isotonic calibration (linear between breakpoints)."""
    model = load_model()
    iso_x = model["iso_x"]
    iso_y = model["iso_y"]

    # Probabilities outside the fitted range take the end values.
    x = min(max(raw_prob, iso_x[0]), iso_x[-1])
    if x <= iso_x[0]:
        return iso_y[0]

    # A linear scan is fine: the export holds a few dozen breakpoints.
    for i in range(1, len(iso_x)):
        if x <= iso_x[i]:
            x0, x1 = iso_x[i - 1], iso_x[i]
            y0, y1 = iso_y[i - 1], iso_y[i]
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return iso_y[-1]


def score_transaction(txn: dict) -> dict:
    """Score one transaction: sum the tree outputs, apply the logistic function, then calibrate.

    Returns the calibrated fraud probability, whether it reaches the operating
    threshold, the threshold, and the share of the sender's balance moved.
    """
    m = load_model()
    features = engineer_features(txn)
    try:
        values = []
        for name in m["feature_names"]:
            values.append(features[name])

        # Each tree returns one leaf value; the margin is the base margin plus their sum.
        leaf_values = []
        for tree in m["trees"]:
            leaf_values.append(eval_tree(tree, values))
        margin = m["base_margin"] + sum(leaf_values)
    except (KeyError, IndexError, TypeError) as exc:
        # A malformed export (missing node key, bad child index) is a server fault, not bad input.
        raise ModelUnavailable(UNAVAILABLE_MESSAGE) from exc
    raw_prob = 1.0 / (1.0 + math.exp(-margin))
    try:
        fraud_probability = isotonic_predict(raw_prob)
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelUnavailable(UNAVAILABLE_MESSAGE) from exc
    return {
        "fraud_probability": fraud_probability,
        "flagged": fraud_probability >= m["operating_threshold"],
        "operating_threshold": m["operating_threshold"],
        "orig_drain_ratio": features["orig_drain_ratio"],
    }


REQUIRED_FIELDS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
MAX_BODY_BYTES = 10_000


def parse_transaction(body: dict) -> dict:
    """Validate a request body and return the five fields as floats.

    Raises ValueError for a missing field, a non-numeric value, NaN, infinity or a
    negative number. Amounts and balances in PaySim are never negative, and the
    pure-Python traversal only matches XGBoost on finite inputs.
    """
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in body:
            missing.append(field)
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    txn = {}
    for field in REQUIRED_FIELDS:
        value = body[field]
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a number")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a number") from None
        if not math.isfinite(number) or number < 0:
            raise ValueError("amount and balances must be finite, non-negative numbers")
        txn[field] = number
    return txn


def check_body_length(length: int) -> None:
    """Reject a request body that is negative in length or longer than MAX_BODY_BYTES."""
    if length < 0 or length > MAX_BODY_BYTES:
        raise ValueError("request body too large")


class handler(BaseHTTPRequestHandler):
    """Vercel serverless handler: POST a JSON transaction to receive its fraud score."""

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: dict, extra_headers: dict = None):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        if extra_headers is None:
            extra_headers = {}
        for key, value in extra_headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        self._send_json(405, {"error": "Use POST with a JSON transaction to get a fraud score"},
                        {"Allow": "POST, OPTIONS"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            check_body_length(length)
            body = json.loads(self.rfile.read(length) or b"{}")
            txn = parse_transaction(body)
            result = score_transaction(txn)
        except ModelUnavailable:
            self._send_json(503, {"error": UNAVAILABLE_MESSAGE}, {"Retry-After": "5"})
            return
        except Exception as exc:  # noqa: BLE001 -- return the error to the caller, not a 500 wall
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, result)
