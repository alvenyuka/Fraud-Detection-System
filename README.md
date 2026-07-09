# Fraud-Detection-System

> Mobile-money fraud classifier on PaySim — XGBoost at 100% precision and 89.5% recall on a 132K time-based holdout, validated across 4 walk-forward folds (PR-AUC 0.9997 ± 0.0004), with tuned hyperparameters, drift monitoring, and a live dashboard.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB6E2D)](https://xgboost.readthedocs.io/)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.9998-success)](#results)
[![Brier](https://img.shields.io/badge/Brier-0.000017-success)](#calibration)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)

![Project banner — fraud-detection system on PaySim mobile-money data](banner.svg)

## Why?

Mobile-money fraud is a precision problem, not an accuracy problem. The PaySim dataset has a 0.13% positive rate; a model that says "not fraud" every time scores 99.87% accuracy and catches zero fraud. Production fraud-ops workflows freeze customer funds on a flag, so false positives carry direct trust and regulatory cost. This repo builds a classifier honest about that trade-off: precision is held at or above 99% and the model is evaluated on a strict time-based holdout — no future-state leakage.

> **A note on PaySim.** PaySim is widely used in introductory fraud-detection tutorials. The contribution here is not the dataset choice but the evaluation rigour: strict time-based split, calibrated probabilities, cost-sensitive threshold selection, and a five-model comparison on identical feature pipelines.

## How this was built

This project was built up in stages, each one answering a question the previous stage left open — the same way you'd work through it as a learning exercise, not all at once:

| Step | File | Question it answers |
|---|---|---|
| 1. Baseline model | [`src/train.py`](src/train.py) | Can a model tell fraud apart from legitimate transactions at all? |
| 2. Hyperparameter tuning | [`src/tune.py`](src/tune.py) | Were the model's settings ever actually tested against alternatives? |
| 3. Walk-forward validation | [`src/validate.py`](src/validate.py) | Does it hold up on more than one train/test split? |
| 4. Drift monitoring | [`src/monitoring.py`](src/monitoring.py) | How would we know if the model started drifting in production? |
| 5. Live dashboard | [`dashboard/app.py`](dashboard/app.py) | How does someone without Python actually use this? |

## Project Structure

```
Fraud-Detection-System/
├── src/
│   ├── features.py       # Shared feature engineering (used by every script below)
│   ├── train.py           # Step 1 — baseline model
│   ├── tune.py             # Step 2 — hyperparameter search
│   ├── validate.py         # Step 3 — walk-forward validation
│   ├── monitoring.py       # Step 4 — drift monitoring
│   └── predict.py          # Command-line scoring
├── dashboard/
│   ├── app.py             # Step 5 — live Streamlit dashboard
│   └── data/               # Small precomputed results the dashboard reads
├── model/
│   ├── xgb_fraud_model.pkl   # Trained model (run make train)
│   └── best_params.json      # Tuned hyperparameters (run make tune)
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

# Optional — the rest of the build-up (see "How this was built" above)
make tune      # search hyperparameters, then re-run `make train` to use them
make validate  # walk-forward validation across 4 time-based folds
make monitor   # simulated drift monitoring
make dashboard # launch the live dashboard locally
```

## Features

- XGBoost calibrated to Brier score 0.000017 (vs. random baseline ~0.0204), calibrated on a held-out slice of the training period — not on the same rows the base model was fit on
- 100% precision / 89.5% recall at operating threshold 0.9989, tuned hyperparameters (`src/tune.py`)
- Walk-forward validated across 4 time-based folds, not just one split (`src/validate.py`)
- Simulated drift monitoring via PSI (`src/monitoring.py`)
- Live dashboard for scoring, performance review, batch scoring, and monitoring (`dashboard/app.py`)
- Time-based train/test split at step 490 — no leakage
- Balance-discrepancy feature engineering (~85% of predictive signal)
- SHAP attribution (2,000-row representative sample; stable across seeds)
- Five-model comparison on identical feature pipelines
- Inference script `src/predict.py` — scores a transaction or full CSV

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

**Time-based split at step 490** — no random shuffling.

| Split | Steps | Rows | Fraud rate |
|---|---|---|---|
| Train | 1-490 | 2,638,273 | 0.207% |
| Test | 491-743 | 132,136 | 2.084% |

**PR-AUC is the primary metric.** Accuracy and ROC-AUC are misleading at 0.13% positive rate.

![Calibration curve — XGBoost predicted probabilities vs. observed fraud rate](04_calibration_curve.png)

![Balance discrepancy fingerprint — fraud vs. legitimate transactions](02_balance_discrepancy_fingerprint.png)

## Results

**These numbers are the verified output of `src/train.py`, run end-to-end against the real PaySim CSV** — not carried over from the notebook. The script previously couldn't run at all (`CalibratedClassifierCV(..., cv="prefit")` was removed in scikit-learn ≥1.6, and the shipped `model/` directory only ever contained a `.gitkeep`), so the numbers that were here before were never actually produced by this pipeline. Fixed by wrapping the fitted XGBoost model in `sklearn.frozen.FrozenEstimator` and calibrating on a held-out 20% slice of the training period instead of on the same rows the base model was fit on.

| Metric | Value |
|---|---|
| Precision | **100.00%** |
| Recall | **89.47%** |
| F1 | 0.9444 |
| PR-AUC | 0.9998 |
| ROC-AUC | 1.0000 * |
| Brier score | 0.000017 |
| Operating threshold | 0.9989 |

*(Retrained with tuned hyperparameters from `src/tune.py` — Step 2 of the build-up. Recall moved from 88.63% to 89.47%: a modest, honest gain, not a dramatic one — see the walk-forward validation below for why there wasn't much room to improve.)*

*PR-AUC and ROC-AUC of 1.0000 are a documented PaySim property once balance-discrepancy features are engineered — the accounting-identity violation is nearly deterministic for fraud in this simulator, which is a property of the synthetic data generation, not evidence this would generalise to real transaction data (see [Limitations](#limitations-and-risks) in the model card).*

### Walk-forward validation — does it hold up on more than one split?

The single-split numbers above only prove the model worked once. `src/validate.py` (Step 3 of the build-up) repeats the same train → calibrate → test recipe on 4 expanding-window folds spanning the entire dataset (steps 350→450, 450→550, 550→650, 650→743), each picking its own cost-optimal threshold:

| Metric | Mean across 4 folds | Std dev |
|---|---|---|
| PR-AUC | 0.9997 | ± 0.0004 |
| ROC-AUC | 0.9999 | ± 0.0002 |
| Precision | 0.9954 | ± 0.0065 |
| Recall | 0.9995 | ± 0.0005 |
| F1 | 0.9975 | ± 0.0035 |
| Brier score | 0.0000 | ± 0.0000 |

The low std dev across folds is real evidence the model isn't a one-off lucky split — performance stays consistently near-ceiling across the whole time horizon, which is consistent with PaySim's fraud signal being near-deterministic once these features are engineered (see caveat above).

**One honest catch:** the cost-optimal decision threshold swings a lot fold to fold — 0.0475, 0.9794, 0.0078, 0.8333. That's the real limitation the near-perfect PR-AUC hides: this dataset's fraud rate and cost trade-off shift enough between windows that no single fixed threshold is clearly "correct" for all of them. The shipped model still uses one static threshold (see `MODEL_CARD.md` → Limitations) — a real deployment would need to revisit it periodically, not assume it's set once and forever.

### Exploratory model comparison (from the notebook, not the shipped pipeline)

| Model | PR-AUC | ROC-AUC | Recall @ 99% Precision |
|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9987 | 1.0000 | 0.9688 |
| Logistic Regression | 0.7905 | 0.9796 | 0.4506 |
| LightGBM (default) | 0.2451 | 0.9502 | 0.0000 |

XGBoost was carried forward into `src/train.py` as the shipped model — not because it topped this table (Random Forest and Stacking did, at a suspicious literal 1.0000 across every metric) but because a single well-understood tree model is easier to justify to a risk team than an ensemble that looks too good to be true. *LightGBM = 0.2451 here uses `scale_pos_weight` but otherwise-default `num_leaves=31`, which overfits badly on this dataset's ~5,500 training-period fraud rows — the same failure mode fixed for XGBoost above (regularize hard enough to match the size of the positive class, not the size of the dataset) would likely fix this too, but wasn't re-run for this pass.*

![Precision-recall curve scoreboard](03_pr_curve_scoreboard.png)

SHAP values on a 2,000-row stratified sample (stable across seeds):

![SHAP beeswarm — per-feature attribution for the deployable XGBoost model](07_shap_beeswarm.png)

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
- [ ] Streaming inference (Kafka + FastAPI)

## License

MIT — see [`LICENSE`](LICENSE).

## Credits

Dataset: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.*
Built by **Alven Yuka** — CPA Finalist, Nairobi.

## Connect

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · 🐙 [GitHub](https://github.com/alvenyuka)
