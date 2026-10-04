"""
business_impact.py: what the shipped model is worth on the holdout, in money and alerts.

On the holdout (TRANSFER and CASH_OUT transactions after step 490), compare three ways
of screening transfers:

  no screening       every fraud goes through
  isFlaggedFraud     PaySim's built-in flag, scored as the column in the data. PaySim
                     documents it as a rule on transfers above 200,000, but the data does
                     not follow that rule: the smallest flagged amount is 353,874.22 and
                     many larger transfers are not flagged (see the notebook).
  shipped model      the calibrated XGBoost at its cost-optimal threshold

For each: the value of fraud stopped, the value that got through, and the number of
honest customers whose transaction was frozen. Amounts are PaySim's simulated currency
units; the value of a fraud is its transaction amount. A second table shows the same
model at a fixed comparison threshold of 0.5, since the cut-off is a business choice
(see README).

Break-even cost per freeze: of the two model thresholds, the lower one flags more. The
extra fraud value it stops, divided by the extra honest customers it freezes, is the
cost per wrongful freeze at which the two thresholds cost the same. If wrongly freezing
one honest customer costs more than this many units, the higher threshold is the
cheaper choice. It is printed and saved as break_even_cost_per_freeze.

Writes dashboard/data/business_impact.json and figures/fraud_value_stopped.png.

Usage
-----
    python src/business_impact.py --data PS_20174392719_1491204439457_log.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, engineer_features, load_and_filter  # noqa: E402
from train import SPLIT_STEP, time_based_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "xgb_fraud_model.pkl"
COMPARISON_THRESHOLD = 0.5


def screening_outcome(flagged, is_fraud, amount) -> dict:
    """Value of fraud stopped and let through, and honest transactions frozen, for one screen."""
    flagged = np.asarray(flagged, dtype=bool)
    is_fraud = np.asarray(is_fraud, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    total_fraud_value = float(amount[is_fraud].sum())
    stopped = float(amount[flagged & is_fraud].sum())
    return {
        "frauds_caught": int((flagged & is_fraud).sum()),
        "frauds_missed": int((~flagged & is_fraud).sum()),
        "fraud_value_stopped": stopped,
        "fraud_value_missed": total_fraud_value - stopped,
        "share_of_fraud_value_stopped": stopped / total_fraud_value if total_fraud_value else 0.0,
        "honest_transactions_frozen": int((flagged & ~is_fraud).sum()),
        "honest_value_frozen": float(amount[flagged & ~is_fraud].sum()),
    }


def break_even(lower: dict, higher: dict) -> dict:
    """Extra value stopped and extra honest customers frozen by the lower threshold over
    the higher one, and the cost per freeze at which the two break even."""
    extra_value = lower["fraud_value_stopped"] - higher["fraud_value_stopped"]
    extra_frozen = lower["honest_transactions_frozen"] - higher["honest_transactions_frozen"]
    return {
        "extra_frauds_caught": lower["frauds_caught"] - higher["frauds_caught"],
        "extra_fraud_value_stopped": extra_value,
        "extra_honest_frozen": extra_frozen,
        "break_even_cost_per_freeze": extra_value / extra_frozen if extra_frozen > 0 else None,
    }


def plot(results: dict, path: Path, n_transactions: int) -> None:
    """Bar chart of fraud value stopped and let through by each screening method on the holdout, saved to `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    stopped = [results[n]["fraud_value_stopped"] / 1e9 for n in names]
    missed = [results[n]["fraud_value_missed"] / 1e9 for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh(names, stopped, color="#2b6cb0", label="Fraud value stopped")
    ax.barh(names, missed, left=stopped, color="#c53030", label="Fraud value let through")
    for i, n in enumerate(names):
        r = results[n]
        ax.text(stopped[i] + missed[i] + 0.05, i,
                f"{r['share_of_fraud_value_stopped']:.2%} stopped, {r['honest_transactions_frozen']:,} honest frozen",
                va="center", fontsize=9, color="#2d3748")
    ax.set_xlabel("Billions of PaySim simulated units")
    ax.set_xlim(0, (stopped[0] + missed[0]) * 1.9)
    ax.invert_yaxis()
    ax.set_title(f"Fraud value on the {n_transactions:,}-transaction holdout, by screening method",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Score the holdout with the shipped model and write the business-impact JSON and chart."""
    p = argparse.ArgumentParser(description="Business impact of the shipped model on the holdout.")
    p.add_argument("--data", required=True, metavar="CSV", help="Path to PaySim CSV.")
    args = p.parse_args()

    artifact = joblib.load(MODEL_PATH)
    threshold = float(artifact["operating_threshold"])
    df = engineer_features(load_and_filter(args.data))
    _, test = time_based_split(df, SPLIT_STEP)
    probs = artifact["model"].predict_proba(test[FEATURE_COLS])[:, 1]
    fraud, amount = test["isFraud"].to_numpy(), test["amount"].to_numpy()

    operating_key = f"Model, threshold {threshold:.4f}"
    comparison_key = f"Model, threshold {COMPARISON_THRESHOLD}"
    results = {
        "No screening": screening_outcome(np.zeros(len(test), bool), fraud, amount),
        "isFlaggedFraud flag": screening_outcome(test["isFlaggedFraud"].to_numpy() == 1, fraud, amount),
        operating_key: screening_outcome(probs >= threshold, fraud, amount),
        comparison_key: screening_outcome(probs >= COMPARISON_THRESHOLD, fraud, amount),
    }
    (lower_t, lower_key), (higher_t, higher_key) = sorted(
        [(threshold, operating_key), (COMPARISON_THRESHOLD, comparison_key)])
    trade_off = {"lower_threshold": lower_t, "higher_threshold": higher_t,
                 **break_even(results[lower_key], results[higher_key])}
    out = ROOT / "dashboard" / "data" / "business_impact.json"
    out.write_text(json.dumps({
        "population": f"{len(test):,} TRANSFER and CASH_OUT transactions after step {SPLIT_STEP}",
        "n_transactions": int(len(test)),
        "currency": "PaySim simulated units; a fraud's value is its transaction amount",
        "operating_threshold": threshold,
        "comparison_threshold": COMPARISON_THRESHOLD,
        "results": results,
        "lower_vs_higher_threshold": trade_off,
    }, indent=2))
    (ROOT / "figures").mkdir(exist_ok=True)
    plot({k: v for k, v in results.items() if k != "No screening"},
         ROOT / "figures" / "fraud_value_stopped.png", len(test))
    print(pd.DataFrame(results).T.to_string())
    print()
    print(f"Lower threshold {lower_t:.4f} against higher threshold {higher_t:.4f}:")
    print(f"  extra frauds caught        {trade_off['extra_frauds_caught']:,}")
    print(f"  extra fraud value stopped  {trade_off['extra_fraud_value_stopped']:,.2f} units")
    print(f"  extra honest customers frozen {trade_off['extra_honest_frozen']:,}")
    be = trade_off["break_even_cost_per_freeze"]
    print("  break_even_cost_per_freeze " + (f"{be:,.2f} units" if be is not None else "undefined (no extra freezes)"))
    print(f"wrote {out.relative_to(ROOT)} and figures/fraud_value_stopped.png")


if __name__ == "__main__":
    main()
