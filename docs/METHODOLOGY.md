# Fraud-Detection-System: full methodology

> The detailed write-up behind the short [README](../README.md): method, every result, tests, and known limitations.

Mobile-money fraud classifier on PaySim, a simulator whose fraud is near-deterministic once these features are engineered, so read the scores as evaluation practice rather than as production performance: XGBoost at 98.42% precision and 99.60% recall on a 132,136-transaction time-based holdout, validated across 4 walk-forward folds (PR-AUC 0.9983 ± 0.0019), with hyperparameters tuned only on earlier data, drift monitoring, SHAP diagnostics and a live dashboard.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB6E2D)](https://xgboost.readthedocs.io/)
[![tests](https://github.com/alvenyuka/Fraud-Detection-System/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Fraud-Detection-System/actions/workflows/ci.yml)

![Project banner: fraud-detection system on PaySim mobile-money data](../banner.svg)

## Live

**[fraud-detection-alven.vercel.app](https://fraud-detection-alven.vercel.app)**: case-study page with the full build-up, verified results, charts, and a live "Score a transaction" form. The scoring form runs the shipped model, re-implemented in pure Python (see `src/export_model_json.py` and `api/score.py`) so the demo is always on with no cold start. `tests/test_export_parity.py` scores 20,000 fixed-seed transactions through both that port and the committed model and asserts the agreement, which is measured at a worst case of about 1.9e-06 in absolute probability and a mean of about 6e-10.

**[Full dashboard](https://fraud-detection-system-kmeuq7hku8tglnxdpmalfk.streamlit.app/)**: the walk-forward results, feature-importance charts, batch CSV scoring, and drift-monitoring timeline, plus a SHAP waterfall for each score. Runs on a free-tier host and may take ~20-30s to wake up on first load. See "How this was built" below for what each tab is backed by.

### Dashboard Preview

| Tab | What it shows |
|---|---|
| Score a Transaction | Fill in one transaction, get a fraud probability, a flag/no-flag verdict, a SHAP waterfall for *why*, and a contextual warning when the score is driven by a full-balance drain (see Step 6 below). |
| Model Performance | Walk-forward validation metrics, PR and calibration curves, feature-importance bar chart, predicted-probability distribution (fraud vs. legitimate), and an interactive threshold slider showing precision/recall/cost at any cutoff (all from `src/explain.py`, Step 7). |
| Batch Scoring | Upload a CSV, get every row scored with live KPI cards (total transactions, fraud rate, high-risk alerts, average amount) and flagged rows highlighted in the results table. |
| Monitoring | PSI drift-over-time chart per engineered feature, against moderate/significant-shift reference lines. |

*(Screenshots aren't checked into the repo. Open the live link above to see it running against the PaySim data.)*

## Question

Mobile-money fraud is mostly a precision problem. The PaySim dataset has a 0.13% positive rate, so a model that says "not fraud" every time scores 99.87% accuracy while catching zero fraud. Production fraud-ops workflows freeze customer funds on a flag, so false positives carry direct trust and regulatory cost. This repo reports precision on a strict time-based holdout, with no row of the training window drawn from after the test window: 98.42% on the single holdout, and 99.0% on average across four walk-forward folds (lowest fold 98.1%). The holdout has a 2.08% fraud rate, about 16 times the dataset's 0.13% overall rate, so precision at production prevalence would be lower.

> **A note on PaySim.** PaySim is a simulator, not a sample of real traffic, and it is widely used in introductory fraud-detection tutorials, so it's a common choice. What this repo adds is the evaluation rigour: strict time-based split, calibrated probabilities, cost-sensitive threshold selection, and a five-model comparison on identical feature pipelines.

## How this was built

This project was built up in stages, each one answering a question the previous stage left open:

| Step | File | What it establishes |
|---|---|---|
| 1. Baseline model | [`src/train.py`](../src/train.py) | Whether a model can separate fraud from legitimate transactions at all |
| 2. Hyperparameter tuning | [`src/tune.py`](../src/tune.py) | Whether the default settings were ever tested against alternatives, or just left as-is |
| 3. Walk-forward validation | [`src/validate.py`](../src/validate.py) | Whether the model holds up on more than one train/test split |
| 4. Drift monitoring | [`src/monitoring.py`](../src/monitoring.py) | How the model's inputs get tracked for drift once it's in production |
| 5. Live dashboard | [`dashboard/app.py`](../dashboard/app.py) | How someone without Python can actually use this |
| 6. Feature fix | [`src/features.py`](../src/features.py) | What live-testing the dashboard turned up: the model was flagging legitimate account closures as 100% fraud, why that happened, and how it was fixed |
| 7. Diagnostics | [`src/explain.py`](../src/explain.py) | Whether one feature dominates the model's decisions, and where the precision/recall/cost trade-off sits as the threshold moves |

**Step 6 in detail:** the model used to take the raw balance columns (`oldbalanceOrg`, `newbalanceOrig`, etc.) as direct inputs alongside the engineered discrepancy features. Because PaySim's simulated fraud almost always drains the sender's account to exactly zero, the model learned "balance hits zero" as a fraud signal on its own: a $12 transaction that fully and correctly emptied a $12 account scored 100% fraud probability, even with zero actual accounting discrepancy. The raw balance columns were removed from the model's inputs (see `MODEL_CARD.md` § Feature Engineering).

Re-testing after applying that fix showed it was only partial. `orig_drain_ratio` (`amount / oldbalanceOrg`) still encodes "was the account fully drained" without needing the raw columns. `src/scenarios.py` reproduces this from the committed model, and writes `dashboard/data/scenario_table.json`, so the numbers below can be regenerated with `make scenarios` and no dataset. On a TRANSFER out of a 10,000 account into a recipient holding 2,000 beforehand, with the recipient credited in full every time:

| Share of the sender's balance moved | Fraud probability |
|---|---|
| 10% / 25% / 50% / 75% / 90% / 99% | between 0.0000% and 0.0102% |
| 100% | 85.00% |

A step function at exactly 100%, with at least three orders of magnitude between any partial drain and the full one. That is PaySim's fraud-generation process showing through the data, not a bug in the feature list. The recipient's opening balance moves these numbers, which is why it is stated: the same full drain with nothing credited to the recipient scores 95.73%. See `MODEL_CARD.md` § Feature Engineering for the full breakdown and why fixing it further would mean retraining on transaction data from a real payment system, rather than removing more columns.

## Project Structure

```
Fraud-Detection-System/
├── src/
│   ├── features.py       # Shared feature engineering (used by every script below)
│   ├── train.py           # Step 1: baseline model
│   ├── tune.py             # Step 2: hyperparameter search
│   ├── validate.py         # Step 3: walk-forward validation
│   ├── monitoring.py       # Step 4: drift monitoring
│   ├── explain.py          # Step 7: feature importance / probability spread / threshold curve
│   ├── scenarios.py        # Regenerates MODEL_CARD's scenario tables from the shipped model
│   ├── export_model_json.py  # Dumps the trained model to plain JSON for api/score.py
│   └── predict.py          # Command-line scoring
├── tests/
│   ├── test_features.py             # Accounting identity + cost-sensitive threshold
│   ├── test_export_parity.py        # api/score.py's port against the shipped model
│   └── test_split_and_monitoring.py # Leakage guard + PSI drift
├── api/
│   └── score.py           # Pure-Python model port for the live Vercel demo
├── dashboard/
│   ├── app.py             # Step 5: live Streamlit dashboard
│   ├── requirements.txt   # Lean dependency set for Streamlit Cloud deployment
│   └── data/               # Small precomputed results the dashboard reads
├── model/
│   ├── xgb_fraud_model.pkl   # Trained model (run make train)
│   ├── best_params.json      # Tuned hyperparameters (run make tune)
│   └── model_export.json     # Dependency-free export used by api/score.py
├── .github/workflows/ci.yml   # Runs the tests on every push
├── Fraud Detection System.ipynb
├── conftest.py             # Puts src/ on sys.path for the tests
├── pytest.ini
├── requirements.txt
├── Makefile
├── MODEL_CARD.md
└── LICENSE
```

## Quick Start

```bash
git clone https://github.com/alvenyuka/Fraud-Detection-System.git
cd Fraud-Detection-System

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make test   # runs in seconds, needs neither the dataset nor a trained model

make train DATA=PS_20174392719_1491204439457_log.csv
make predict

# Optional: the rest of the build-up (see "How this was built" above)
make tune      # search hyperparameters, then re-run `make train` to use them
make validate  # walk-forward validation across 4 time-based folds
make monitor   # simulated drift monitoring
make dashboard # launch the live dashboard locally
```

**What each step costs.** `make test` needs nothing but the repo. Everything
below it needs the 470MB PaySim CSV, and the notebook is the heavy one:

| Step | Needs the dataset | Rough cost |
|---|---|---|
| `make test` | no | seconds |
| `make train` | yes | a few minutes |
| `make tune` | yes | about 50 minutes, 40 trials |
| `make validate` | yes | about 8 minutes, four walk-forward folds |
| the notebook, end to end | yes | 45 to 90 minutes, and roughly 6 to 8GB of free memory |

The notebook is heavier than the pipeline because it fits five models rather
than one, including a stacking ensemble that refits three base learners across
three cross-validation folds. Thread counts are bounded by `N_JOBS` in the
config cell for that reason: `n_jobs=-1` nests inside the stack, and each
worker takes its own copy of a 2.6-million-row frame, so the peak is set by how
many are alive at once rather than by the data. Raise `N_JOBS` if you have the
headroom.

## Features

- XGBoost calibrated to Brier score 0.000138 (vs. random baseline ~0.0204), calibrated on a held-out slice of the training period, not on the same rows the base model was fit on
- 98.42% precision / 99.60% recall at operating threshold 0.0123, hyperparameters tuned on windows that precede every reporting split (`src/tune.py`), threshold picked dynamically by cost (see `src/features.py::pick_best_threshold`) rather than hardcoded
- Walk-forward validated across 4 time-based folds, not just one split (`src/validate.py`)
- Simulated drift monitoring via PSI (`src/monitoring.py`)
- Feature importance, probability spread, and threshold/cost trade-off diagnostics (`src/explain.py`)
- Live dashboard for scoring, performance review, batch scoring, and monitoring (`dashboard/app.py`)
- Time-based train/test split at step 490, so no training row comes from after the test window
- Balance-discrepancy feature engineering. The sender side dominates: `orig_drain_ratio` carries 40.0% of mean |SHAP| attribution and `orig_balance_discrepancy` 23.1%, so the two sender-side features carry 63.0% between them (`dashboard/data/feature_importance.json`)
- SHAP attribution (2,000-row representative sample; stable across seeds)
- Five-model comparison on identical feature pipelines
- Inference script `src/predict.py`: scores a transaction or full CSV
- Always-on live scoring demo (`api/score.py`), a pure-Python port whose agreement with the real model is enforced by a test rather than asserted in prose

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Modelling | `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn` |
| Explainability | `shap` |
| Serialisation | `joblib` |
| Testing | `pytest`, GitHub Actions |
| Environment | `jupyter`, `jupyterlab` |

## Installation

```bash
git clone https://github.com/alvenyuka/Fraud-Detection-System.git
cd Fraud-Detection-System
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download PaySim from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place `PS_20174392719_1491204439457_log.csv` in the project root.

## Usage

```bash
jupyter lab "Fraud Detection System.ipynb"   # full notebook
make train                                    # train + serialise model
make predict                                  # score one transaction (interactive)
make predict-csv INPUT=txns.csv OUTPUT=out.csv
```

## Dataset

| Property | Value |
|---|---|
| Rows | 6,362,620 |
| Fraud rate | 0.1291% |
| Active fraud types | TRANSFER, CASH_OUT only |
| Filtered dataset | 2,770,409 rows |

## Methodology

**Time-based split at step 490.** No random shuffling.

| Split | Steps | Rows | Fraud rate |
|---|---|---|---|
| Train | 1-490 | 2,638,273 | 0.207% |
| Test | 491-743 | 132,136 | 2.084% |

**PR-AUC is the primary metric.** Accuracy and ROC-AUC are misleading at 0.13% positive rate.

![Calibration curve: XGBoost predicted probabilities vs. observed fraud rate](../04_calibration_curve.png)

![Balance discrepancy fingerprint: fraud vs. legitimate transactions](../02_balance_discrepancy_fingerprint.png)

## Results

These numbers are the output of `src/train.py`, run end to end against the full PaySim CSV. (An earlier version of the script could not run at all, because `CalibratedClassifierCV(..., cv="prefit")` was removed in scikit-learn 1.6; it now wraps the fitted model in `sklearn.frozen.FrozenEstimator` and calibrates on a held-out 20% of the training period.)

| Metric | Value |
|---|---|
| Precision | **98.42%** |
| Recall | **99.60%** |
| F1 | 0.9901 |
| PR-AUC | 0.9996 |
| ROC-AUC | 1.0000 (rounded) |
| Brier score | 0.000138 |
| Operating threshold | 0.0123 |

The threshold is chosen on the calibration split by `src/features.py::pick_best_threshold`, costing a missed fraud at 100 times a false alarm. Because the calibrated probabilities reflect a 0.21% training fraud rate, the cost-minimising cutoff sits far below 0.5.

### Hyperparameter tuning

`src/tune.py` runs 40 Optuna trials. Each is scored by mean PR-AUC over three windows that all end by step 350 (train up to step 200, 250 or 300, score the following 50 steps), with early stopping on the last fifth of each window's training steps. No row used to choose the settings is used to report a result: walk-forward testing starts at step 351 and the holdout at step 491. The search chose 500 trees of depth 4, learning rate 0.20, row subsample 0.60 and column subsample 0.81 (`model/best_params.json`).

### Walk-forward validation: does it hold up on more than one split?

`src/validate.py` repeats the same train, calibrate and test recipe on 4 expanding-window folds (testing steps 351-450, 451-550, 551-650 and 651-743). Each fold chooses its threshold on its own calibration split, never on its test rows.

| Metric | Mean across 4 folds | Std dev |
|---|---|---|
| PR-AUC | 0.9983 | ± 0.0019 |
| ROC-AUC | 0.9998 | ± 0.0002 |
| Precision | 0.9898 | ± 0.0083 |
| Recall | 0.9973 | ± 0.0022 |
| F1 | 0.9936 | ± 0.0037 |
| Brier score | 0.0001 | ± 0.0001 |

The weakest fold is the first (PR-AUC 0.9952, precision 0.9813). The standard deviations use `np.std`'s population estimator on four folds, so the per-fold values in `dashboard/data/walk_forward_results.csv` are the more useful thing to read.

**What changed from earlier versions.** Earlier walk-forward figures chose each fold's threshold on that fold's test labels, which pushed recall to exactly 1.0 in three folds, and used hyperparameters tuned on windows that overlapped folds 1 and 2 and a quarter of the holdout. Both were fixed and everything was re-run; the holdout precision moved from 99.85% to 98.42%, which is the honest figure.

The cost-optimal threshold still swings from fold to fold (0.50, 0.89, 0.019, 0.017), which the near-perfect PR-AUC hides: the fraud rate and cost trade-off shift enough between windows that no single fixed threshold is clearly right for all of them. A production deployment would need to revisit the threshold on a schedule rather than set it once.

### Exploratory model comparison (from the notebook, not the shipped pipeline)

| Model | PR-AUC | ROC-AUC | Recall @ 99% Precision |
|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9987 | 1.0000 | 0.9688 |
| Logistic Regression | 0.7905 | 0.9796 | 0.4506 |
| LightGBM (default) | 0.2451 | 0.9502 | 0.0000 |

XGBoost was carried forward into `src/train.py` as the shipped model. It didn't top this table (Random Forest and Stacking did, at a suspicious literal 1.0000 across every metric). It was chosen because a single well-understood tree model is easier to justify to a risk team than an ensemble that looks too good to be true. *LightGBM = 0.2451 here uses `scale_pos_weight` but otherwise-default `num_leaves=31`, which overfits badly on this dataset's ~5,500 training-period fraud rows. The same failure mode fixed for XGBoost above (regularize hard enough to match the size of the positive class, rather than the size of the dataset) would likely fix this too, but wasn't re-run for this pass.*

![Precision-recall curve scoreboard](../03_pr_curve_scoreboard.png)

SHAP values on a 2,000-row stratified sample (stable across seeds):

![SHAP beeswarm: per-feature attribution for the deployable XGBoost model](../07_shap_beeswarm.png)

## Known Limitations

PaySim is a simulator. The near-1.0 PR-AUC above comes from how deterministic its fraud-generation process becomes once these features are engineered, so it isn't evidence this generalises to production traffic (see `MODEL_CARD.md`). The drain-ratio artifact described in "How this was built" is only partially fixed: `orig_drain_ratio` still leaks the "fully-drained account" fraud signature, and closing that gap fully would mean retraining on transaction data from a real payment system rather than dropping more columns. The shipped model also uses one static decision threshold, even though the cost-optimal threshold swings meaningfully fold to fold in walk-forward validation, so a production deployment would need to revisit it periodically.

Three more, each of which changes how a number above should be read:

- **This is a sender-side full-drain detector, not a general fraud detector.** `dest_balance_discrepancy` carries 6.2% of SHAP attribution, so the model barely reacts to whether the recipient actually received the money: the same full drain scores 95.73% whether 0%, 25%, 50% or 75% of the debited amount arrives (`make scenarios`). A fraud that skims half an account without draining it, with nothing reaching the recipient, scores 0.0071%. `MODEL_CARD.md` has the full sweep and the rule-based fix that was tried and rejected.
- **One of the model's five inputs has drifted heavily over the period it is scored on.** `src/monitoring.py` reports PSI of 1.4163 for `dest_balance_discrepancy`, 5.7 times the 0.25 "investigate" line, past that line in 14 of 15 windows, with a median of 0.79. `orig_drain_ratio` reaches 0.7813 in the final window. The shipped model trains on steps 1 to 490 and runs one static threshold, so this is a real finding about it, not a demonstration of the monitoring code. The walk-forward folds retrain on everything up to each split and so never feel that drift.
- **Thread count is part of the model.** XGBoost draws its row-subsample mask from per-thread RNG streams, so with `subsample` below 1.0 the fitted model depends on how many threads it ran with, regardless of `random_state` (measured on a 60,000-row synthetic frame: up to 0.093 apart in predicted probability between one thread and four at `subsample=0.84`; column subsampling does not cause it). `n_jobs` is fixed at 4 in training, tuning and validation, so re-running `make tune` and `make train` reproduces these figures; changing it moves them slightly.

## Tests

```bash
make test        # or: python -m pytest
```

28 tests, about a minute and a half, run in CI on Python 3.11 and 3.13 on every
push. None needs the 470MB PaySim CSV: each one builds a small frame by hand or
from a fixed seed. That is deliberate. The point is not to re-check the model's
score, which a walk-forward run already reports, but to pin the pieces of logic
that can break silently and still leave every downstream number looking
plausible. The parity test does load the committed model artifacts, which is
what makes it the slow one.

| What is tested | Why it is the thing that can break silently |
|---|---|
| **The time-based split** (`time_based_split`) | Fraud data is temporal. A random split lets the model see later transactions from accounts it is later asked to score, which inflates every metric downstream, and nothing downstream can tell a leaked score from an earned one. One test asserts the training window ends strictly before the test window begins; another shuffles the same frame to demonstrate the overlap the guard prevents. |
| **The accounting identity** (`engineer_features`) | The balance-discrepancy features come from arithmetic, not from learning, so they can be asserted exactly rather than approximately. A clean transaction must score zero on both sides; a drained-origin transaction must break the identity by exactly the amount. Zero balances are checked separately, because an `inf` or `NaN` from a zero denominator would quietly become a category of its own inside the model. |
| **The cost-sensitive threshold** (`pick_best_threshold`) | A model that ranks perfectly and cuts at the wrong threshold still loses money. The test sweeps its own independent grid of cutoffs, rather than the candidate set the implementation builds, and asserts none beats the one returned; a second test pins the 100-to-1 cost ratio that makes the optimiser choose a threshold well below 0.5. |
| **PSI drift** (`calculate_psi`) | PSI is the number that decides whether the model has gone stale in production, so it is asserted to be zero on identical samples, to grow monotonically as a distribution shifts, and to stay finite on a constant feature rather than raising and stopping the whole monitoring run. |
| **The pure-Python port** (`api/score.py`) | The live demo re-implements tree traversal and isotonic calibration by hand, with no xgboost and no scikit-learn, so a wrong default-branch rule or an off-by-one in the isotonic scan would change scores without raising anything and the demo would keep serving plausible numbers. The test scores 20,000 fixed-seed transactions through both paths and asserts a bound it measures rather than one quoted from memory. |

`FEATURE_COLS` membership is also pinned by name, and the raw balance columns are
asserted absent. The test that iterates the list would otherwise delete its own
coverage the moment an entry was removed, and the raw balances are exactly the
shortcut Step 6 removed.

Training, tuning, walk-forward validation and drift monitoring all need the
dataset, so they stay local steps behind the Makefile rather than running in CI.

## Roadmap

- [x] Time-based evaluation harness
- [x] Five-model comparison
- [x] Calibration + SHAP attribution
- [x] Inference script (`src/predict.py`)
- [x] Training pipeline (`src/train.py`)
- [x] Model card (`MODEL_CARD.md`)
- [x] Hyperparameter tuning (`src/tune.py`)
- [x] Walk-forward validation across multiple time splits (`src/validate.py`)
- [x] Drift monitoring (PSI on balance-discrepancy features, `src/monitoring.py`)
- [x] Live dashboard (`dashboard/app.py`)
- [x] Feature-importance / threshold-cost diagnostics (`src/explain.py`)
- [x] Always-on live scoring demo (`api/score.py`, `src/export_model_json.py`)
- [x] Unit tests on the leakage guard, the accounting identity, the threshold search and PSI, run in CI (`tests/`)
- [ ] Streaming inference (Kafka + FastAPI)

## License

MIT. See [`LICENSE`](../LICENSE).

## Credits

Dataset: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.*
Built by **Alven Yuka**, CPA Finalist.
