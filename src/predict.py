"""
predict.py — the command-line way to use the model train.py produces.

Loads the saved model and scores one or more transactions. This is the
plain-Python entry point; dashboard/app.py (Step 5 of the build-up) wraps
this same scoring logic in a web page for people who don't want to use the
command line.

Usage — score a single transaction (interactive)
-------------------------------------------------
    python src/predict.py --model model/xgb_fraud_model.pkl

Usage — score a CSV of transactions
-------------------------------------
    python src/predict.py \\
        --model model/xgb_fraud_model.pkl \\
        --input transactions.csv \\
        --output scored.csv

CSV format expected (header required, column order flexible):
    type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest

Only TRANSFER and CASH_OUT rows produce a fraud score. Other types are passed
through with fraud_score=NaN and fraud_flag=0.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import ACTIVE_TYPES, FEATURE_COLS, engineer_features  # noqa: E402

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> dict:
    """Load the serialised model artifact produced by train.py."""
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(
            f"Unexpected artifact format in {model_path}. "
            "Re-run src/train.py to regenerate."
        )
    return artifact


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_dataframe(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """
    Add fraud_score and fraud_flag columns to df.

    Only TRANSFER and CASH_OUT rows are scored; all others get NaN / 0.
    """
    model     = artifact["model"]
    threshold = artifact["operating_threshold"]

    df = df.copy()
    df["fraud_score"] = np.nan
    df["fraud_flag"]  = 0

    mask = df["type"].isin(ACTIVE_TYPES) if "type" in df.columns else pd.Series(True, index=df.index)
    active = engineer_features(df[mask])

    probs = model.predict_proba(active[FEATURE_COLS])[:, 1]
    df.loc[mask, "fraud_score"] = probs
    df.loc[mask, "fraud_flag"]  = (probs >= threshold).astype(int)

    return df


def score_single(artifact: dict) -> None:
    """Interactive mode: prompt user for transaction fields, print result."""
    threshold = artifact["operating_threshold"]

    print("\n-- Transaction details --")
    txn_type = input("  type (TRANSFER / CASH_OUT): ").strip().upper()
    if txn_type not in ACTIVE_TYPES:
        print(f"  Type '{txn_type}' is not in the fraud-active set. Score: N/A")
        return

    try:
        amount   = float(input("  amount: "))
        old_org  = float(input("  oldbalanceOrg: "))
        new_org  = float(input("  newbalanceOrig: "))
        old_dest = float(input("  oldbalanceDest: "))
        new_dest = float(input("  newbalanceDest: "))
    except ValueError:
        print("  Error: all balance/amount fields must be numeric.", file=sys.stderr)
        sys.exit(1)

    row = pd.DataFrame([{
        "type":           txn_type,
        "amount":         amount,
        "oldbalanceOrg":  old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
    }])

    scored  = score_dataframe(row, artifact)
    score   = scored["fraud_score"].iloc[0]
    flag    = int(scored["fraud_flag"].iloc[0])

    verdict = "FRAUD FLAGGED" if flag else "LEGITIMATE"
    print("\n-- Result --")
    print(f"  Fraud probability : {score:.6f}")
    print(f"  Operating threshold: {threshold}")
    print(f"  Decision          : {verdict}")
    print("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score transactions with the trained XGBoost fraud classifier."
    )
    p.add_argument(
        "--model",
        default="model/xgb_fraud_model.pkl",
        metavar="PKL",
        help="Path to serialised model artifact (default: model/xgb_fraud_model.pkl).",
    )
    p.add_argument(
        "--input",
        metavar="CSV",
        help="CSV of transactions to score. If omitted, runs in interactive mode.",
    )
    p.add_argument(
        "--output",
        metavar="CSV",
        help="Output CSV with fraud_score and fraud_flag columns appended.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"Error: model not found at '{model_path}'.\n"
            "Run 'make train' to train the model first.",
            file=sys.stderr,
        )
        sys.exit(1)

    artifact = load_model(str(model_path))

    if args.input:
        df = pd.read_csv(args.input)
        scored = score_dataframe(df, artifact)

        flagged = int(scored["fraud_flag"].sum())
        print(f"Scored {len(scored):,} transactions -> {flagged:,} flagged ({100*flagged/len(scored):.2f}%)")

        if args.output:
            scored.to_csv(args.output, index=False)
            print(f"Results saved -> {args.output}")
        else:
            flagged_rows = scored[scored["fraud_flag"] == 1]
            if len(flagged_rows):
                print("\nFlagged transactions:")
                print(flagged_rows[["type", "amount", "fraud_score", "fraud_flag"]].to_string(index=False))
    else:
        score_single(artifact)


if __name__ == "__main__":
    main()
