# Fraud-Detection-System: full methodology

> The detailed write-up behind the short [README](../README.md): method, every result, tests, and known limitations.

Mobile-money fraud classifier on PaySim, a simulator whose fraud is near-deterministic once these features are engineered, so read the scores as evaluation practice rather than as production performance: XGBoost at 99.89% precision and 99.56% recall on a 132,136-transaction time-based holdout, validated across 4 walk-forward folds (PR-AUC 0.9971 ± 0.0020, precision 0.9690 ± 0.0508), with hyperparameters tuned only on earlier data, drift monitoring, SHAP diagnostics and a live dashboard.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-EB6E2D)](https://xgboost.readthedocs.io/)
[![tests](https://github.com/alvenyuka/Fraud-Detection-System/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Fraud-Detection-System/actions/workflows/ci.yml)

![Project banner: fraud-detection system on PaySim mobile-money data](../banner.svg)

## Live

**[fraud-detection-alven.vercel.app](https://fraud-detection-alven.vercel.app)**: case-study page with the full build-up, verified results, charts, and a live "Score a transaction" form. The scoring form runs the shipped model, re-implemented in pure Python (see `src/export_model_json.py` and `api/score.py`) so the demo is always on with no cold start. `tests/test_export_parity.py` scores 20,000 fixed-seed transactions through both that port and the committed model and asserts the agreement, which for the model trained on 2026-10-02 is measured at a worst case of about 6.6e-08 in absolute probability and a mean of about 1.4e-10. The function loads `model/model_export.json` from the `main` branch when it starts, so pushing a new export to `main` changes the live scores without a redeploy. It rejects missing, non-numeric, NaN, infinite and negative inputs and bodies over 10,000 bytes.

**[Full dashboard](https://fraud-detection-system-kmeuq7hku8tglnxdpmalfk.streamlit.app/)**: the walk-forward results, feature-importance charts, batch CSV scoring, and drift-monitoring timeline, plus a SHAP waterfall for each score. Runs on a free-tier host and may take ~20-30s to wake up on first load. See "How this was built" below for what each tab is backed by.

### Dashboard Preview

| Tab | What it shows |
|---|---|
| Score a Transaction | Fill in one transaction, get a fraud probability, a flag/no-flag verdict, a SHAP waterfall for *why*, and a contextual warning when the score is driven by a full-balance drain (see Step 6 below). |
| Model Performance | Walk-forward validation metrics, PR and calibration curves, feature-importance bar chart, predicted-probability distribution (fraud vs. legitimate), and an interactive threshold slider showing precision/recall/cost at any cutoff (all from `src/explain.py`, Step 7). |
| Batch Scoring | Upload a CSV, get every row scored with live KPI cards (total transactions, flagged share, high-risk alerts, average amount) and flagged rows highlighted in the results table. Files with missing columns, negative or non-numeric values, or no rows are rejected with a message. |
| Monitoring | PSI drift-over-time chart per engineered feature, against moderate/significant-shift reference lines. |

*(Screenshots aren't checked into the repo. Open the live link above to see it running against the PaySim data.)*

## Question

Mobile-money fraud is mostly a precision problem. The PaySim dataset has a 0.13% positive rate, so a model that says "not fraud" every time scores 99.87% accuracy while catching zero fraud. Production fraud-ops workflows freeze customer funds on a flag, so false positives carry direct trust and regulatory cost. This repo reports precision on a strict time-based holdout, with no row of the training window drawn from after the test window: 99.89% on the single holdout, and 96.9% on average across four walk-forward folds (lowest fold 88.1%). The holdout has a 2.08% fraud rate, about 16 times the dataset's 0.13% overall rate, so precision at production prevalence would be lower.

> **A note on PaySim.** PaySim is a simulator, not a sample of real traffic, and it is widely used in introductory fraud-detection tutorials, so it's a common choice. What this repo adds is the evaluation rigour: strict time-based split, calibrated probabilities, cost-sensitive threshold selection, and a five-model comparison on the notebook's own feature set.

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

**Step 6 in detail:** the model used to take the raw balance columns (`oldbalanceOrg`, `newbalanceOrig`, etc.) as direct inputs alongside the engineered discrepancy features. Because PaySim's simulated fraud almost always drains the sender's account to exactly zero, the model learned "balance hits zero" as a fraud signal on its own: a 12-unit transaction that fully and correctly emptied a 12-unit account scored 100% fraud probability, even with zero actual accounting discrepancy. The raw balance columns were removed from the model's inputs (see `MODEL_CARD.md` § Feature Engineering).

Re-testing after applying that fix showed it was only partial. `orig_drain_ratio` (`amount / oldbalanceOrg`) still encodes "was the account fully drained" without needing the raw columns. `src/scenarios.py` reproduces this from the committed model, and writes `dashboard/data/scenario_table.json`, so the numbers below can be regenerated with `make scenarios` and no dataset. On a TRANSFER out of a 10,000 account into a recipient holding 2,000 beforehand, with the recipient credited in full every time:

| Share of the sender's balance moved | Fraud probability |
|---|---|
| 10% / 25% / 50% / 75% / 90% / 99% | 0.0000% |
| 100% | 90.24% |

A step function at exactly 100%: every partial drain scores zero to four decimal places and the full one 90.24%, above the 0.8852 operating threshold. A 12-unit balance moved in full scores 88.52%, at the threshold and so flagged. That is PaySim's fraud-generation process showing through the data, not a bug in the feature list. The recipient's opening balance moves these numbers, which is why it is stated: the same full drain with nothing credited to the recipient scores 99.18%. See `MODEL_CARD.md` § Feature Engineering for the full breakdown and why fixing it further would mean retraining on transaction data from a real payment system, rather than removing more columns.

## Project Structure

```
Fraud-Detection-System/
├── src/
│   ├── config.py          # Split step, tuning windows and walk-forward folds
│   ├── features.py       # Shared feature engineering (used by every script below)
│   ├── train.py           # Step 1: baseline model
│   ├── tune.py             # Step 2: hyperparameter search
│   ├── validate.py         # Step 3: walk-forward validation
│   ├── monitoring.py       # Step 4: drift monitoring
│   ├── explain.py          # Step 7: feature importance / probability spread / threshold curve
│   ├── scenarios.py        # Regenerates MODEL_CARD's scenario tables from the shipped model
│   ├── business_impact.py  # Fraud value stopped by each screen, and the break-even cost per freeze
│   ├── make_figures.py     # Redraws the README charts from dashboard/data
│   ├── export_model_json.py  # Dumps the trained model to plain JSON for api/score.py
│   └── predict.py          # Command-line scoring
├── tests/
│   ├── test_features.py             # Accounting identity + cost-sensitive threshold
│   ├── test_export_parity.py        # api/score.py's port against the shipped model
│   ├── test_api_validation.py       # api/score.py rejects NaN, inf, negatives, big bodies
│   ├── test_pipeline_guards.py      # Both fixed leaks, calibration split, predict.py
│   ├── test_business_impact.py      # Value-stopped arithmetic
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
├── Fraud_Detection_System.ipynb
├── conftest.py             # Puts src/ on sys.path for the tests
├── pytest.ini
├── requirements.txt        # Exact pins of the training environment
├── requirements-lock.txt   # Full dependency lock
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
| `make train` | yes | about a minute |
| `make tune` | yes | 40 trials; the 2026-10-02 run took 4 hours on a machine shared with other jobs |
| `make validate` | yes | under two minutes, four walk-forward folds |
| the notebook, end to end | yes | 45 to 90 minutes, and roughly 6 to 8GB of free memory |

The notebook is heavier than the pipeline because it fits five models rather
than one, including a stacking ensemble that refits three base learners across
three cross-validation folds. Thread counts are bounded by `N_JOBS` in the
config cell for that reason: `n_jobs=-1` nests inside the stack, and each
worker takes its own copy of a 2.6-million-row frame, so the peak is set by how
many are alive at once rather than by the data. Raise `N_JOBS` if you have the
headroom.

## Features

- XGBoost calibrated to Brier score 0.000131 (a model that always predicts the 2.08% holdout base rate scores about 0.0204), calibrated on the latest fifth of the training steps, not on the rows the base model was fit on
- 99.89% precision / 99.56% recall at operating threshold 0.8852, hyperparameters and tree count tuned on windows that precede every reporting split (`src/tune.py`), threshold picked dynamically by cost (see `src/features.py::pick_best_threshold`) rather than hardcoded
- Walk-forward validated across 4 time-based folds, not just one split (`src/validate.py`)
- Simulated drift monitoring via PSI (`src/monitoring.py`)
- Feature importance, probability spread, and threshold/cost trade-off diagnostics (`src/explain.py`)
- Live dashboard for scoring, performance review, batch scoring, and monitoring (`dashboard/app.py`)
- Time-based train/test split at step 490, so no training row comes from after the test window
- Balance-discrepancy feature engineering. The sender side dominates: `orig_balance_discrepancy` carries 51.8% of mean |SHAP| attribution and `orig_drain_ratio` 25.1%, so the two sender-side features carry 76.9% between them (`dashboard/data/feature_importance.json`)
- SHAP attribution (2,000-row representative sample; stable across seeds)
- Five-model comparison in the notebook, on the notebook's own feature set
- Inference script `src/predict.py`: scores a transaction or full CSV
- Always-on live scoring demo (`api/score.py`), a pure-Python port whose agreement with the real model is enforced by a test rather than asserted in prose

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12+ (exact library pins in `requirements.txt`, full lock in `requirements-lock.txt`) |
| Modelling | `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn` |
| Explainability | `shap` |
| Serialisation | `joblib` |
| Testing | `pytest`, GitHub Actions |
| Environment | `nbconvert`, `ipykernel` (and `jupyterlab` to edit the notebook) |

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
jupyter lab Fraud_Detection_System.ipynb   # full notebook
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

![Balance discrepancy fingerprint: fraud vs. legitimate transactions](../figures/balance_discrepancy_fingerprint.png)

## Results

These numbers are the output of `src/train.py`, run end to end against the full PaySim CSV on 2026-10-02. (An earlier version of the script could not run at all, because `CalibratedClassifierCV(..., cv="prefit")` was removed in scikit-learn 1.6; it now wraps the fitted model in `sklearn.frozen.FrozenEstimator`.) The base model is fit on steps 1 to 392 and calibrated on steps 393 to 490, the latest fifth of the training period.

| Metric | Value |
|---|---|
| Precision | **99.89%** |
| Recall | **99.56%** (2,742 of 2,754 frauds; 3 false alarms) |
| F1 | 0.9973 |
| PR-AUC | 0.9984 |
| ROC-AUC | 1.0000 (rounded) |
| Brier score | 0.000131 |
| Operating threshold | 0.8852 |

The threshold is chosen on the calibration split by `src/features.py::pick_best_threshold`, costing a missed fraud at 100 times a false alarm. It is high because the 35-tree model with isotonic calibration produces only 12 distinct scores: on the holdout every cut-off from 0.01 to 0.88 makes the same decisions, and on a cost tie the search keeps the highest tied threshold, which raises the fewest alerts.

### Hyperparameter tuning

`src/tune.py` runs 40 Optuna trials. Each is scored by mean PR-AUC over three windows that all end by step 350 (train up to step 200, 250 or 300, score the following 50 steps), with early stopping on the last fifth of each window's training steps. No row used to choose the settings is used to report a result: walk-forward testing starts at step 351 and the holdout at step 491. The search chose depth 4, learning rate 0.20, row subsample 0.60, column subsample 0.81 and `scale_pos_weight` 2.14 (`model/best_params.json`). Early stopping kept 63, 35 and 11 trees in the winning trial's three windows, and the shipped model uses the median, 35, so it is the configuration that was scored. Earlier versions shipped 500 trees, a count no trial had been evaluated with.

### Walk-forward validation: does it hold up on more than one split?

`src/validate.py` repeats the same train, calibrate and test recipe on 4 expanding-window folds (testing steps 351-450, 451-550, 551-650 and 651-743). Each fold calibrates and chooses its threshold on the latest fifth of its own training steps, never on its test rows.

| Metric | Mean across 4 folds | Std dev |
|---|---|---|
| PR-AUC | 0.9971 | ± 0.0020 |
| ROC-AUC | 0.9989 | ± 0.0013 |
| Precision | 0.9690 | ± 0.0508 |
| Recall | 0.9971 | ± 0.0029 |
| F1 | 0.9821 | ± 0.0262 |
| Brier score | 0.000115 | ± 0.000076 |

| Fold (test steps) | Calibration steps | Threshold | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|
| 351-450 | 281-350 | 1.0000 | 0.9974 | 1.0000 | 0.9948 |
| 451-550 | 361-450 | 0.8182 | 0.9996 | 0.9960 | 1.0000 |
| 551-650 | 441-550 | 0.9855 | 0.9939 | 0.9991 | 0.9938 |
| 651-743 | 521-650 | 0.0210 | 0.9974 | 0.8810 | 1.0000 |

The weakest fold by ranking is the third (PR-AUC 0.9939); the weakest by precision is the fourth (0.8810), where the threshold chosen on the calibration steps was far lower than in the other folds. The standard deviations use `np.std`'s population estimator on four folds, so the per-fold values in `dashboard/data/walk_forward_results.csv` are the more useful thing to read.

**What changed from earlier versions.** Earlier walk-forward figures chose each fold's threshold on that fold's test labels, which pushed recall to exactly 1.0 in three folds, and used hyperparameters tuned on windows that overlapped folds 1 and 2 and a quarter of the holdout. Both were fixed and everything was re-run; the holdout precision moved from 99.85% to 98.42%. The retrain of 2026-10-02 shipped the evaluated 35-tree configuration and moved calibration from a random 20% of the training rows to the latest 20% of its steps: holdout precision moved from 98.42% to 99.89%, PR-AUC from 0.9996 to 0.9984, and walk-forward precision from 0.9898 to 0.9690.

The cost-optimal threshold swings from fold to fold (min 0.0210, max 1.0000, coefficient of variation 0.57 in `dashboard/data/metrics_summary.json`), which the near-perfect PR-AUC hides. Moving to a time-ordered calibration slice did not narrow the range; with only a handful of distinct calibrated scores, a small change in the calibration window can move the cost-optimal cut-off from one score level to another. A production deployment would need to revisit the threshold on a schedule rather than set it once.

### Exploratory model comparison (from the notebook, not the shipped pipeline)

The notebook compares five models on its own raw-plus-error-balance features, on the same time split
(train to step 490, test after it). From its full run on 30 September 2026:

| Model | PR-AUC | ROC-AUC | Recall @ 99% precision |
|---|---|---|---|
| Stacking ensemble | 1.0000 | 1.0000 | 1.0000 |
| Random forest | 1.0000 | 1.0000 | 1.0000 |
| XGBoost + SMOTE (ablation) | 0.9995 | 1.0000 | 0.9949 |
| Logistic regression | 0.7901 | 0.9795 | 0.4503 |
| XGBoost, `scale_pos_weight` 336 | 0.4771 | 0.9855 | 0.0000 |
| LightGBM, default settings | 0.0357 | 0.7041 | 0.0000 |

The notebook's earlier stored outputs showed its XGBoost at 0.9987; they predated the removal of `step` (the
split variable) from its features. Without `step`, a shallow XGBoost weighted by the raw class ratio (336)
over-fits the few fraud rows and assigns probability 1.0 to legitimate transactions on the later window, while
the same model trained on SMOTE-balanced data reaches 0.9995. The shipped pipeline treats the class weight as
a hyperparameter: `src/tune.py`, choosing only on earlier data, picked 2.14. XGBoost is still the shipped model
because, with tuned settings and the engineered balance features, it matches the ensembles' ranking (holdout
PR-AUC 0.9996) as a single, explainable model, and the ensembles' literal 1.0000 on a simulator is more likely
a sign of PaySim's determinism than of a better model. LightGBM's `num_leaves=31` default over-fits the
training period's roughly 5,500 fraud rows in the same way.

The notebook renders the precision-recall scoreboard, calibration, permutation-importance and SHAP charts for these models inline.

![Holdout precision and recall by threshold, and the four walk-forward folds, for the shipped model](../figures/threshold_tradeoff.png)

## Known Limitations

PaySim is a simulator. The near-1.0 PR-AUC above comes from how deterministic its fraud-generation process becomes once these features are engineered, so it isn't evidence this generalises to production traffic (see `MODEL_CARD.md`). The drain-ratio artifact described in "How this was built" is only partially fixed: `orig_drain_ratio` still leaks the "fully-drained account" fraud signature, and closing that gap fully would mean retraining on transaction data from a real payment system rather than dropping more columns. The shipped model also uses one static decision threshold, even though the cost-optimal threshold swings meaningfully fold to fold in walk-forward validation, so a production deployment would need to revisit it periodically.

Three more, each of which changes how a number above should be read:

- **This is a sender-side full-drain detector, not a general fraud detector.** `dest_balance_discrepancy` carries 3.9% of SHAP attribution, so the model barely reacts to whether the recipient actually received the money: the same full drain scores 99.18% whether 0%, 25%, 50% or 75% of the debited amount arrives (`make scenarios`). A fraud that skims half an account without draining it, with nothing reaching the recipient, scores 0.0000%. `MODEL_CARD.md` has the full sweep and the rule-based fix that was tried and rejected.
- **Two of the model's inputs drift past the "investigate" line in the final hours.** `src/monitoring.py` measures PSI against the shipped model's training window (steps 1 to 490). In the last window (steps 701-743) `orig_drain_ratio` reaches 0.7642 and `dest_amount_ratio` 0.2635, both past 0.25; `dest_balance_discrepancy` stays between 0.10 and 0.25 in every window from step 451 on. Earlier versions measured PSI against steps 1 to 50 instead, which is why they reported larger values for `dest_balance_discrepancy`. The shipped model runs one static threshold, so this is a finding about it, not a demonstration of the monitoring code.
- **Thread count is part of the model.** XGBoost draws its row-subsample mask from per-thread RNG streams, so with `subsample` below 1.0 the fitted model depends on how many threads it ran with, regardless of `random_state` (measured on a 60,000-row synthetic frame: up to 0.093 apart in predicted probability between one thread and four at `subsample=0.84`; column subsampling does not cause it). `n_jobs` is fixed at 4 in training, tuning and validation, so re-running `make tune` and `make train` reproduces these figures; changing it moves them slightly.

## Tests

```bash
make test        # or: python -m pytest
```

55 tests, a couple of minutes, run in CI on Python 3.12 and 3.13 on every push
(one, which needs optuna, is skipped there). CI installs the exact versions pinned
in `requirements.txt`, so the committed pickle is loaded by the library versions
that wrote it. None needs the 470MB PaySim CSV: each one builds a small frame by hand or
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
| **Both leaks this repo once had** (`tests/test_pipeline_guards.py`) | Every tuning window must end before every reporting window, checked against the same `src/config.py` the scripts use; and `run_one_fold` must hand `pick_best_threshold` the calibration labels, checked by recording the length and fraud count of what it receives. Either leak would raise every reported metric without breaking anything. |
| **The calibration split and the scoring entry point** (`split_off_latest_steps`, `predict.score_dataframe`, `predict.load_model`) | Calibration rows must be the latest steps; scoring must use the feature list saved with the model, mask non-fraud types, reject missing columns, and warn when the library versions differ from the ones that wrote the pickle. |
| **API input validation** (`api/score.py`) | XGBoost sends NaN down each split's default branch while the port sends it right, so the API must refuse NaN, infinities, negatives and oversized bodies before scoring. |
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
