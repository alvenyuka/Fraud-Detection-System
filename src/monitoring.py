"""
monitoring.py — Step 4 of the model build-up: drift monitoring.

A model that scores well today can quietly stop working later if the data
it sees in production starts looking different from the data it was trained
on — this is called "drift". This project's own README listed drift
monitoring as something not yet built. This script builds a simple version
of it.

We don't have real production traffic to monitor, so we simulate it: PaySim
already spans many time steps, so we treat the earliest slice of time as
"what the model was trained on" and check how much every later slice has
drifted away from it.

The tool used to measure drift is PSI (Population Stability Index) — a single
number per feature per time window:
    PSI < 0.10        -> stable, no action needed
    0.10 <= PSI < 0.25 -> moderate shift, worth watching
    PSI >= 0.25        -> significant shift, investigate

Usage
-----
    python src/monitoring.py --data PS_20174392719_1491204439457_log.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import engineer_features, load_and_filter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WINDOW_SIZE_STEPS = 50  # roughly 2 days of PaySim time per window
N_BINS = 10

# The 4 engineered features carry almost all of the model's predictive
# signal (see MODEL_CARD.md), so drift here matters most.
MONITORED_FEATURES = [
    "orig_balance_discrepancy",
    "dest_balance_discrepancy",
    "orig_drain_ratio",
    "dest_amount_ratio",
]

DASHBOARD_DATA_DIR = Path(__file__).parent.parent / "dashboard" / "data"


def calculate_psi(reference_values: np.ndarray, current_values: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Compare two samples of the same feature and return one drift number.

    Steps:
      1. Cut the reference sample into `n_bins` equal-sized groups (deciles by default).
      2. Count what fraction of the reference sample falls in each group — this is the
         "expected" shape of the distribution.
      3. Count what fraction of the current sample falls in the *same* groups — the
         "actual" shape now.
      4. PSI adds up how far "actual" has drifted from "expected", bin by bin.
    A result near 0 means the two samples look the same; a large result means the
    feature's distribution has shifted.
    """
    bin_edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, n_bins + 1)))
    if len(bin_edges) < 3:
        return 0.0  # feature barely varies — nothing meaningful to compare

    bin_edges[0], bin_edges[-1] = -np.inf, np.inf  # catch any values outside the reference range
    reference_counts, _ = np.histogram(reference_values, bins=bin_edges)
    current_counts, _ = np.histogram(current_values, bins=bin_edges)

    reference_fraction = np.maximum(reference_counts / max(len(reference_values), 1), 1e-6)
    current_fraction = np.maximum(current_counts / max(len(current_values), 1), 1e-6)

    return float(np.sum((current_fraction - reference_fraction) * np.log(current_fraction / reference_fraction)))


def monitor(data_path: str) -> None:
    df = load_and_filter(data_path)
    df = engineer_features(df)

    min_step, max_step = int(df["step"].min()), int(df["step"].max())
    window_start_steps = list(range(min_step, max_step, WINDOW_SIZE_STEPS))
    log.info(
        "Splitting steps %d-%d into %d windows of %d steps each ...",
        min_step, max_step, len(window_start_steps), WINDOW_SIZE_STEPS,
    )

    # The earliest window stands in for "the data the model was trained on".
    reference_window = df[df["step"] < min_step + WINDOW_SIZE_STEPS]
    log.info("Reference window: steps %d-%d (%d rows)", min_step, min_step + WINDOW_SIZE_STEPS - 1, len(reference_window))

    psi_rows = []
    for window_start in window_start_steps:
        window_end = window_start + WINDOW_SIZE_STEPS
        current_window = df[(df["step"] >= window_start) & (df["step"] < window_end)]
        if len(current_window) == 0:
            continue

        for feature in MONITORED_FEATURES:
            drift_score = calculate_psi(reference_window[feature].to_numpy(), current_window[feature].to_numpy())
            psi_rows.append({
                "window_start_step": window_start,
                "window_end_step": window_end,
                "feature": feature,
                "psi": round(drift_score, 4),
                "n_rows": len(current_window),
            })

    psi_timeline = pd.DataFrame(psi_rows)
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DASHBOARD_DATA_DIR / "psi_timeline.csv"
    psi_timeline.to_csv(output_path, index=False)

    log.info("Highest PSI reached per feature across the whole time span:")
    worst_psi_per_feature = psi_timeline.groupby("feature")["psi"].max().round(4)
    for feature, worst_psi in worst_psi_per_feature.items():
        verdict = "significant shift" if worst_psi > 0.25 else ("moderate shift" if worst_psi > 0.1 else "stable")
        log.info("  %-28s worst PSI=%.4f (%s)", feature, worst_psi, verdict)

    log.info("Drift monitoring timeline written -> %s", output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSI drift monitoring for the fraud classifier's features.")
    p.add_argument("--data", required=True, metavar="CSV", help="Path to PaySim CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    monitor(args.data)
