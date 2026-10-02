"""
scenarios.py: the pre-deployment scenario sweeps MODEL_CARD.md quotes.

Those tables used to be numbers in a document with no script behind them, which
is the same failure mode README section Results describes catching once before.
This script regenerates them from the committed model artifact and writes
dashboard/data/scenario_table.json, so the card quotes a file rather than a
memory. It needs model/xgb_fraud_model.pkl and nothing else: no dataset, no
training run.

The construction matters and was not previously stated. Every scenario below is
a TRANSFER from an account holding 10,000, to a recipient account holding 2,000
beforehand. The recipient's opening balance is not a detail: the score moves
with it, so a table that omits it cannot be reproduced. 2,000 is used because
it is an unremarkable mid-range recipient balance, not because it flatters
anything.

Two sweeps:

  diversion  the sender is drained to zero, and a varying share of the
             debited amount is credited to the recipient. This is the sweep
             that shows the model barely notices whether the money arrived.

  drain      the recipient is credited in full every time, and the share of
             the sender's balance being moved varies. This is the sweep that
             shows the step function at exactly 100%.

Usage
-----
    python src/scenarios.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, engineer_features  # noqa: E402

MODEL_PATH = Path(__file__).parent.parent / "model" / "xgb_fraud_model.pkl"
OUTPUT_PATH = Path(__file__).parent.parent / "dashboard" / "data" / "scenario_table.json"

SENDER_BALANCE = 10_000.0
RECIPIENT_OPENING_BALANCE = 2_000.0

CREDITED_SHARES = [0.0, 0.25, 0.50, 0.75, 1.00]
DRAIN_SHARES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]


def _score(model, rows: list[dict]) -> list[float]:
    frame = engineer_features(pd.DataFrame(rows))
    return [float(p) for p in model.predict_proba(frame[FEATURE_COLS])[:, 1]]


def diversion_sweep(model) -> list[dict]:
    """Sender fully drained; vary how much of it reaches the recipient."""
    rows = [
        {
            "amount": SENDER_BALANCE,
            "oldbalanceOrg": SENDER_BALANCE,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": RECIPIENT_OPENING_BALANCE,
            "newbalanceDest": RECIPIENT_OPENING_BALANCE + SENDER_BALANCE * share,
        }
        for share in CREDITED_SHARES
    ]
    return [
        {"credited_share": share, "fraud_probability": prob}
        for share, prob in zip(CREDITED_SHARES, _score(model, rows))
    ]


def drain_sweep(model) -> list[dict]:
    """Recipient credited in full; vary how much of the sender's balance moves."""
    rows = [
        {
            "amount": SENDER_BALANCE * share,
            "oldbalanceOrg": SENDER_BALANCE,
            "newbalanceOrig": SENDER_BALANCE * (1 - share),
            "oldbalanceDest": RECIPIENT_OPENING_BALANCE,
            "newbalanceDest": RECIPIENT_OPENING_BALANCE + SENDER_BALANCE * share,
        }
        for share in DRAIN_SHARES
    ]
    return [
        {"drain_share": share, "fraud_probability": prob}
        for share, prob in zip(DRAIN_SHARES, _score(model, rows))
    ]


def partial_skim(model) -> dict:
    """Half the balance moved, none of it credited: the pattern the model misses."""
    row = {
        "amount": SENDER_BALANCE / 2,
        "oldbalanceOrg": SENDER_BALANCE,
        "newbalanceOrig": SENDER_BALANCE / 2,
        "oldbalanceDest": RECIPIENT_OPENING_BALANCE,
        "newbalanceDest": RECIPIENT_OPENING_BALANCE,
    }
    return {"description": "half the balance moved, nothing credited",
            "fraud_probability": _score(model, [row])[0]}


def small_full_drain(model) -> dict:
    """A customer moving their whole 12-unit balance, credited in full: the demo case
    that first exposed the full-drain shortcut."""
    row = {
        "amount": 12.0,
        "oldbalanceOrg": 12.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": RECIPIENT_OPENING_BALANCE,
        "newbalanceDest": RECIPIENT_OPENING_BALANCE + 12.0,
    }
    return {"description": "12-unit balance moved in full, recipient credited in full",
            "fraud_probability": _score(model, [row])[0]}


def main() -> Path:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    payload = {
        "construction": {
            "type": "TRANSFER",
            "sender_opening_balance": SENDER_BALANCE,
            "recipient_opening_balance": RECIPIENT_OPENING_BALANCE,
            "note": (
                "The recipient's opening balance changes the score, so it is "
                "recorded here. Without it the tables below cannot be reproduced."
            ),
        },
        "operating_threshold": bundle["operating_threshold"],
        "diversion_sweep": diversion_sweep(model),
        "drain_sweep": drain_sweep(model),
        "partial_skim": partial_skim(model),
        "small_full_drain": small_full_drain(model),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Sender drained to zero, {RECIPIENT_OPENING_BALANCE:,.0f} in the recipient account:")
    for row in payload["diversion_sweep"]:
        print(f"  {row['credited_share']:>6.0%} credited -> {row['fraud_probability']:.4%}")
    print("\nRecipient credited in full, varying the share of the sender's balance moved:")
    for row in payload["drain_sweep"]:
        print(f"  {row['drain_share']:>6.0%} drained  -> {row['fraud_probability']:.4%}")
    print(f"\n{payload['partial_skim']['description']}: "
          f"{payload['partial_skim']['fraud_probability']:.4%}")
    threshold = payload["operating_threshold"]
    small = payload["small_full_drain"]
    flagged = "flagged" if small["fraud_probability"] >= threshold else "not flagged"
    print(f"{small['description']}: {small['fraud_probability']:.4%} ({flagged} at {threshold:.4f})")
    print()
    print(f"wrote {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
