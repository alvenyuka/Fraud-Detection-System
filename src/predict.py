"""
predict.py: the command-line way to use the model train.py produces.

Loads the saved model and scores one or more transactions. This is the
plain-Python entry point; dashboard/app.py (Step 5 of the build-up) wraps
this same scoring logic in a web page for people who don't want to use the
command line.

Usage, score a single transaction (interactive)
-------------------------------------------------
    python src/predict.py --model model/xgb_fraud_model.pkl

Usage, score a CSV of transactions
-------------------------------------
    python src/predict.py \\
        --model model/xgb_fraud_model.pkl \\
        --input transactions.csv \\
        --output scored.csv

CSV format expected (header required, column order flexible):
    type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest

Only TRANSFER and CASH_OUT rows produce a fraud score. Other types are passed
through with fraud_score=NaN and fraud_flag=0. A file without a `type` column is
rejected unless --assume-active is given, because the model is undefined on the
other transaction types.
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import ACTIVE_TYPES, FEATURE_COLS, engineer_features  # noqa: E402

INPUT_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

# The libraries whose version changes how the pickle loads or scores.
CHECKED_LIBRARIES = ("scikit-learn", "xgboost")


def installed_versions() -> dict:
    """Installed versions of the libraries a saved model depends on, to compare with the versions it was trained on."""
    import sklearn
    import xgboost

    return {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def version_mismatches(artifact: dict) -> list[str]:
    """Libraries whose installed version differs from the one that wrote the artifact."""
    recorded = artifact.get("versions")
    if not recorded:
        return ["artifact records no library versions"]
    installed = installed_versions()
    return [
        f"{lib}: trained with {recorded.get(lib)}, installed {installed[lib]}"
        for lib in CHECKED_LIBRARIES
        if recorded.get(lib) != installed[lib]
    ]


def load_model(model_path: str, strict: bool = False) -> dict:
    """
    Load the serialised model artifact produced by train.py.

    The artifact records the library versions that wrote it. If scikit-learn or
    xgboost differ from those, scores may differ too, so this warns (or raises,
    with strict=True). requirements-lock.txt reproduces the training environment.
    """
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(
            f"Unexpected artifact format in {model_path}. "
            "Re-run src/train.py to regenerate."
        )
    mismatches = version_mismatches(artifact)
    if mismatches:
        message = (
            f"{model_path} was written under different library versions ("
            + "; ".join(mismatches)
            + "). Install requirements-lock.txt to reproduce the shipped scores."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
    return artifact


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_dataframe(df: pd.DataFrame, artifact: dict, assume_active: bool = False) -> pd.DataFrame:
    """
    Add fraud_score and fraud_flag columns to df.

    Only TRANSFER and CASH_OUT rows are scored; all others get NaN / 0. The
    `type` column is required unless assume_active=True, which scores every row
    as if it were one of those two types. The model is scored on the feature
    list saved with it, not on whatever this module currently declares.
    """
    model     = artifact["model"]
    threshold = artifact["operating_threshold"]
    feature_cols = list(artifact.get("feature_cols", FEATURE_COLS))

    missing = [c for c in INPUT_COLS if c not in df.columns]
    if "type" not in df.columns and not assume_active:
        missing.append("type")
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(missing)}")

    df = df.copy()
    df["fraud_score"] = np.nan
    df["fraud_flag"]  = 0

    mask = df["type"].isin(ACTIVE_TYPES) if "type" in df.columns else pd.Series(True, index=df.index)
    if mask.any():
        active = engineer_features(df[mask])
        probs = model.predict_proba(active[feature_cols])[:, 1]
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
    p.add_argument(
        "--assume-active",
        action="store_true",
        help="Score every row when the CSV has no `type` column (treat all rows as TRANSFER/CASH_OUT).",
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
        try:
            scored = score_dataframe(df, artifact, assume_active=args.assume_active)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        flagged = int(scored["fraud_flag"].sum())
        share = 100 * flagged / len(scored) if len(scored) else 0.0
        print(f"Scored {len(scored):,} transactions -> {flagged:,} flagged ({share:.2f}%)")

        if args.output:
            scored.to_csv(args.output, index=False)
            print(f"Results saved -> {args.output}")
        else:
            flagged_rows = scored[scored["fraud_flag"] == 1]
            if len(flagged_rows):
                print("\nFlagged transactions:")
                shown = [c for c in ("type", "amount", "fraud_score", "fraud_flag") if c in flagged_rows.columns]
                print(flagged_rows[shown].to_string(index=False))
    else:
        score_single(artifact)


if __name__ == "__main__":
    main()
