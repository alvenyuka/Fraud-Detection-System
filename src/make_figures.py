"""
make_figures.py: draw the README charts from the pipeline's own outputs.

Reads only dashboard/data/, which src/explain.py and src/validate.py write, so
the charts always describe the shipped model and need neither the dataset nor
a training run.

Writes figures/threshold_tradeoff.png: precision and recall on the holdout as
the decision threshold moves, next to the four walk-forward folds.

Usage
-----
    python src/make_figures.py
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dashboard" / "data"
OUT = ROOT / "figures"
MODEL_PATH = ROOT / "model" / "xgb_fraud_model.pkl"

BLUE, RED, GREY = "#2b6cb0", "#c53030", "#718096"


def main() -> None:
    curve = pd.read_csv(DATA / "threshold_cost_curve.csv")
    folds = pd.read_csv(DATA / "walk_forward_results.csv")
    operating = float(joblib.load(MODEL_PATH)["operating_threshold"])
    curve = curve[curve["threshold"] > 0]  # at 0 every transaction is flagged

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})

    left.plot(curve["threshold"], curve["precision"] * 100, color=BLUE, lw=2, label="Precision")
    left.plot(curve["threshold"], curve["recall"] * 100, color=RED, lw=2, label="Recall")
    left.axvline(operating, color=GREY, ls="--", lw=1)
    left.annotate(f"operating threshold {operating:.4f}\n(cost-optimal at 100:1)",
                  xy=(operating, 70), xytext=(0.12, 72), fontsize=9, color="#2d3748",
                  arrowprops={"arrowstyle": "->", "color": GREY})
    left.set_xlabel("Decision threshold")
    left.set_ylabel("%")
    left.set_ylim(60, 101)
    left.set_title("Holdout: 132,136 transactions after step 490", fontsize=11, loc="left")
    left.legend(frameon=False, loc="lower left")

    labels = [f"{a + 1}-{b}" for a, b in zip(folds["train_end_step"], folds["test_end_step"])]
    x = range(len(folds))
    width = 0.27
    for offset, col, colour in ((-width, "PR-AUC", GREY), (0, "Precision", BLUE), (width, "Recall", RED)):
        right.bar([i + offset for i in x], folds[col] * 100, width, color=colour, label=col)
    right.set_xticks(list(x))
    right.set_xticklabels(labels)
    right.set_xlabel("Walk-forward folds: test window (steps)")
    right.set_ylim(97, 100.2)
    right.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=9)

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "threshold_tradeoff.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
