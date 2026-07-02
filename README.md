# Fraud-Detection-System

> Mobile-money fraud classifier on PaySim — XGBoost at 100% precision and 88.6% recall on a 132K time-based holdout, verified by running `src/train.py` end-to-end against the real dataset.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB6E2D)](https://xgboost.readthedocs.io/)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-1.0000-success)](#results)
[![Brier](https://img.shields.io/badge/Brier-0.000033-success)](#calibration)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)

![Project banner — fraud-detection system on PaySim mobile-money data](banner.svg)

## Why?

Mobile-money fraud is a precision problem, not an accuracy problem. The PaySim dataset has a 0.13% positive rate; a model that says "not fraud" every time scores 99.87% accuracy and catches zero fraud. Production fraud-ops workflows freeze customer funds on a flag, so false positives carry direct trust and regulatory cost. This repo builds a classifier honest about that trade-off: precision is held at or above 99% and the model is evaluated on a strict time-based holdout — no future-state leakage.

> **A note on PaySim.** PaySim is widely used in introductory fraud-detection tutorials. The contribution here is not the dataset choice but the evaluation rigour: strict time-based split, calibrated probabilities, cost-sensitive threshold selection, and a five-model comparison on identical feature pipelines.

## Project Structure

```
Fraud-Detection-System/
├── src/
│   ├── train.py          # Training pipeline — data -> model serialisation
│   └── predict.py        # Inference script — load model, score transactions
├── model/
│   └── xgb_fraud_model.pkl   # Serialised trained model (run make train)
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
```

## Features

- XGBoost calibrated to Brier score 0.000033 (vs. random baseline ~0.0204), calibrated on a held-out slice of the training period — not on the same rows the base model was fit on
- 100% precision / 88.6% recall at operating threshold 0.9989
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
| Recall | **88.63%** |
| F1 | 0.9397 |
| PR-AUC | 1.0000 |
| ROC-AUC | 1.0000 * |
| Brier score | 0.000033 |
| Operating threshold | 0.9989 |

*PR-AUC and ROC-AUC of 1.0000 are a documented PaySim property once balance-discrepancy features are engineered — the accounting-identity violation is nearly deterministic for fraud in this simulator, which is a property of the synthetic data generation, not evidence this would generalise to real transaction data (see [Limitations](#limitations-and-risks) in the model card). At the fixed 0.9989 threshold, precision reaches 100% but recall (88.6%) is 8 points below what the exploratory notebook comparison below reports — the two use different feature sets and aren't directly comparable; this table is the one that reflects what actually ships.*

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
- [ ] Streaming inference (Kafka + FastAPI)
- [ ] Drift monitoring (PSI on balance-discrepancy features)

## License

MIT — see [`LICENSE`](LICENSE).

## Credits

Dataset: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.*
Built by **Alven Yuka** — CPA Finalist (Kenya), awaiting ICPAK membership, Nairobi.

## Connect

[alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · [GitHub](https://github.com/alvenyuka)
