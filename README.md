# Fraud-Detection-System

> Mobile-money fraud classifier on PaySim: XGBoost at 99.85% precision and 99.56% recall on a 132K time-based holdout, validated across 4 walk-forward folds (PR-AUC 0.9986 ± 0.0013), with tuned hyperparameters, drift monitoring, feature-importance/threshold diagnostics, and a live dashboard.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB6E2D)](https://xgboost.readthedocs.io/)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.9993-success)](#results)
[![Brier](https://img.shields.io/badge/Brier-0.00017-success)](#calibration)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)

![Project banner: fraud-detection system on PaySim mobile-money data](banner.svg)

## 🔴 Live

**[fraud-detection-alven.vercel.app](https://fraud-detection-alven.vercel.app)**: case-study page with the full build-up, verified results, charts, and a live "Score a transaction" form. The scoring form runs the shipped model, re-implemented in pure Python (see `src/export_model_json.py` and `api/score.py`) so the demo is always on with no cold start. Its output matches the original scikit-learn/XGBoost pipeline to within floating-point precision on 20,000 held-out transactions.

**[Full dashboard](https://fraud-detection-system-kmeuq7hku8tglnxdpmalfk.streamlit.app/)**: the walk-forward results, feature-importance charts, batch CSV scoring, and drift-monitoring timeline, plus a SHAP waterfall for each score. Runs on a free-tier host and may take ~20-30s to wake up on first load. See "How this was built" below for what each tab is backed by.

### Dashboard Preview

| Tab | What it shows |
|---|---|
| 🔍 Score a Transaction | Fill in one transaction, get a fraud probability, a flag/no-flag verdict, a SHAP waterfall for *why*, and a contextual warning when the score is driven by a full-balance drain (see Step 6 below). |
| 📊 Model Performance | Walk-forward validation metrics, PR and calibration curves, feature-importance bar chart, predicted-probability distribution (fraud vs. legitimate), and an interactive threshold slider showing precision/recall/cost at any cutoff (all from `src/explain.py`, Step 7). |
| 📁 Batch Scoring | Upload a CSV, get every row scored with live KPI cards (total transactions, fraud rate, high-risk alerts, average amount) and flagged rows highlighted in the results table. |
| 📈 Monitoring | PSI drift-over-time chart per engineered feature, against moderate/significant-shift reference lines. |

*(Screenshots aren't checked into the repo. Open the live link above to see it running against real data.)*

## Why?

Mobile-money fraud is mostly a precision problem. The PaySim dataset has a 0.13% positive rate, so a model that says "not fraud" every time scores 99.87% accuracy while catching zero fraud. Production fraud-ops workflows freeze customer funds on a flag, so false positives carry direct trust and regulatory cost. This repo reports precision on a strict time-based holdout, with no future-state leakage: 99.85% on the single holdout, and 95.6% on average across four walk-forward folds (lowest fold 87.3%). The holdout has a 2.08% fraud rate, about 16 times the dataset's 0.13% overall rate, so precision at production prevalence would be lower.

> **A note on PaySim.** PaySim is widely used in introductory fraud-detection tutorials, so it's a common choice. What this repo adds is the evaluation rigour: strict time-based split, calibrated probabilities, cost-sensitive threshold selection, and a five-model comparison on identical feature pipelines.

## How this was built

This project was built up in stages, each one answering a question the previous stage left open:

| Step | File | What it establishes |
|---|---|---|
| 1. Baseline model | [`src/train.py`](src/train.py) | Whether a model can separate fraud from legitimate transactions at all |
| 2. Hyperparameter tuning | [`src/tune.py`](src/tune.py) | Whether the default settings were ever tested against alternatives, or just left as-is |
| 3. Walk-forward validation | [`src/validate.py`](src/validate.py) | Whether the model holds up on more than one train/test split |
| 4. Drift monitoring | [`src/monitoring.py`](src/monitoring.py) | How the model's inputs get tracked for drift once it's in production |
| 5. Live dashboard | [`dashboard/app.py`](dashboard/app.py) | How someone without Python can actually use this |
| 6. Feature fix | [`src/features.py`](src/features.py) | What live-testing the dashboard turned up: the model was flagging legitimate account closures as 100% fraud, why that happened, and how it was fixed |
| 7. Diagnostics | [`src/explain.py`](src/explain.py) | Whether one feature dominates the model's decisions, and where the precision/recall/cost trade-off sits as the threshold moves |

**Step 6 in detail:** the model used to take the raw balance columns (`oldbalanceOrg`, `newbalanceOrig`, etc.) as direct inputs alongside the engineered discrepancy features. Because PaySim's simulated fraud almost always drains the sender's account to exactly zero, the model learned "balance hits zero" as a fraud signal on its own: a $12 transaction that fully and correctly emptied a $12 account scored 100% fraud probability, even with zero actual accounting discrepancy. The raw balance columns were removed from the model's inputs (see `MODEL_CARD.md` § Feature Engineering).

Re-testing after applying that fix showed it was only partial. `orig_drain_ratio` (`amount / oldbalanceOrg`) still encodes "was the account fully drained" without needing the raw columns: a 100%-drained, fully consistent transaction still scores 95.3%, while the exact same transaction at any drain fraction from 10-99% scores a flat 0.006%. That step-function jump at exactly 100% is PaySim's fraud-generation process showing through the data, not a bug in the feature list. See `MODEL_CARD.md` § Feature Engineering for the full breakdown and why fixing it further would mean retraining on transaction data from a real payment system, rather than removing more columns.

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
│   └── predict.py          # Command-line scoring
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
├── Fraud Detection System.ipynb
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

make train DATA=PS_20174392719_1491204439457_log.csv
make predict

# Optional: the rest of the build-up (see "How this was built" above)
make tune      # search hyperparameters, then re-run `make train` to use them
make validate  # walk-forward validation across 4 time-based folds
make monitor   # simulated drift monitoring
make dashboard # launch the live dashboard locally
```

## Features

- XGBoost calibrated to Brier score 0.00017 (vs. random baseline ~0.0204), calibrated on a held-out slice of the training period, not on the same rows the base model was fit on
- 99.85% precision / 99.56% recall at operating threshold 0.4000, tuned hyperparameters (`src/tune.py`), threshold picked dynamically by cost (see `src/features.py::pick_best_threshold`) rather than hardcoded
- Walk-forward validated across 4 time-based folds, not just one split (`src/validate.py`)
- Simulated drift monitoring via PSI (`src/monitoring.py`)
- Feature importance, probability spread, and threshold/cost trade-off diagnostics (`src/explain.py`)
- Live dashboard for scoring, performance review, batch scoring, and monitoring (`dashboard/app.py`)
- Time-based train/test split at step 490, no leakage
- Balance-discrepancy feature engineering (~43% of predictive signal; no single feature dominates, see `MODEL_CARD.md`)
- SHAP attribution (2,000-row representative sample; stable across seeds)
- Five-model comparison on identical feature pipelines
- Inference script `src/predict.py`: scores a transaction or full CSV
- Always-on live scoring demo (`api/score.py`), a pure-Python port validated against the real model to within floating-point precision

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Modelling | `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn` |
| Explainability | `shap` |
| Serialisation | `joblib` |
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

![Calibration curve: XGBoost predicted probabilities vs. observed fraud rate](04_calibration_curve.png)

![Balance discrepancy fingerprint: fraud vs. legitimate transactions](02_balance_discrepancy_fingerprint.png)

## Results

**These numbers are the verified output of `src/train.py`, run end-to-end against the real PaySim CSV**, not carried over from the notebook. The script previously couldn't run at all (`CalibratedClassifierCV(..., cv="prefit")` was removed in scikit-learn ≥1.6, and the shipped `model/` directory only ever contained a `.gitkeep`), so the numbers that were here before were never actually produced by this pipeline. Fixed by wrapping the fitted XGBoost model in `sklearn.frozen.FrozenEstimator` and calibrating on a held-out 20% slice of the training period instead of on the same rows the base model was fit on.

| Metric | Value |
|---|---|
| Precision | **99.85%** |
| Recall | **99.56%** |
| F1 | 0.9971 |
| PR-AUC | 0.9993 |
| ROC-AUC | 0.9998 |
| Brier score | 0.00017 |
| Operating threshold | 0.4000 |

*(These are the numbers from the current shipped model: tuned hyperparameters from `src/tune.py`, raw balance columns removed per Step 6 below, and the operating threshold picked dynamically on the calibration split by `src/features.py::pick_best_threshold` rather than hardcoded. Precision and recall both moved after Step 6's fix, see the walk-forward section below for the full trade-off, and `MODEL_CARD.md` for why a near-1.0 PR-AUC on PaySim isn't evidence this generalises to transaction data from a real payment system.)*

### Walk-forward validation: does it hold up on more than one split?

The single-split numbers above only prove the model worked once. `src/validate.py` (Step 3 of the build-up) repeats the same train → calibrate → test recipe on 4 expanding-window folds spanning the entire dataset (steps 350→450, 450→550, 550→650, 650→743), each picking its own cost-optimal threshold:

| Metric | Mean across 4 folds | Std dev |
|---|---|---|
| PR-AUC | 0.9986 | ± 0.0013 |
| ROC-AUC | 0.9999 | ± 0.0002 |
| Precision | 0.9561 | ± 0.0490 |
| Recall | 0.9998 | ± 0.0004 |
| F1 | 0.9768 | ± 0.0262 |
| Brier score | 0.0002 | ± 0.0001 |

PR-AUC and recall are stable across folds (standard deviations 0.0013 and 0.0004), so the model is not a one-off lucky split. Precision is less stable: it averages 0.9561 with a standard deviation of 0.0490, and the earliest fold (steps 350 to 450, 0.25% fraud) reaches only 0.8731. PR-AUC and recall stay near-ceiling across the time horizon, consistent with PaySim's fraud signal being near-deterministic once these features are engineered (see caveat above).

*(These numbers are from the corrected model, see "How this was built" § Step 6 above. Precision dropped from 0.9954 to 0.9561 and its fold-to-fold variance grew (± 0.0490) after removing the raw balance columns that used to let the model take a shortcut; recall improved slightly. That's the cost of no longer letting the model key off "balance hits zero": a small trade-off for a model that no longer calls legitimate account closures certain fraud.)*

The cost-optimal decision threshold still swings a lot fold to fold, which the near-perfect PR-AUC hides: this dataset's fraud rate and cost trade-off shift enough between windows that no single fixed threshold is clearly "correct" for all of them. The shipped model still uses one static threshold (see `MODEL_CARD.md` → Limitations). A deployment running this in production would need to revisit that threshold periodically rather than set it once.

### Exploratory model comparison (from the notebook, not the shipped pipeline)

| Model | PR-AUC | ROC-AUC | Recall @ 99% Precision |
|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9987 | 1.0000 | 0.9688 |
| Logistic Regression | 0.7905 | 0.9796 | 0.4506 |
| LightGBM (default) | 0.2451 | 0.9502 | 0.0000 |

XGBoost was carried forward into `src/train.py` as the shipped model. It didn't top this table (Random Forest and Stacking did, at a suspicious literal 1.0000 across every metric). It was chosen because a single well-understood tree model is easier to justify to a risk team than an ensemble that looks too good to be true. *LightGBM = 0.2451 here uses `scale_pos_weight` but otherwise-default `num_leaves=31`, which overfits badly on this dataset's ~5,500 training-period fraud rows. The same failure mode fixed for XGBoost above (regularize hard enough to match the size of the positive class, rather than the size of the dataset) would likely fix this too, but wasn't re-run for this pass.*

![Precision-recall curve scoreboard](03_pr_curve_scoreboard.png)

SHAP values on a 2,000-row stratified sample (stable across seeds):

![SHAP beeswarm: per-feature attribution for the deployable XGBoost model](07_shap_beeswarm.png)

## Known Limitations

PaySim is a simulator. The near-1.0 PR-AUC above comes from how deterministic its fraud-generation process becomes once these features are engineered, so it isn't evidence this generalises to production traffic (see `MODEL_CARD.md`). The drain-ratio artifact described in "How this was built" is only partially fixed: `orig_drain_ratio` still leaks the "fully-drained account" fraud signature, and closing that gap fully would mean retraining on transaction data from a real payment system rather than dropping more columns. The shipped model also uses one static decision threshold, even though the cost-optimal threshold swings meaningfully fold to fold in walk-forward validation, so a production deployment would need to revisit it periodically.

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
- [ ] Streaming inference (Kafka + FastAPI)

## License

MIT. See [`LICENSE`](LICENSE).

## Credits

Dataset: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.*
Built by **Alven Yuka**, CPA Finalist.

Every result reported above is the verified output of re-running `src/train.py` end-to-end against the real PaySim dataset; see Results above for the case where that check caught numbers the shipped pipeline had never actually produced.

## Connect

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · 🐙 [GitHub](https://github.com/alvenyuka)
