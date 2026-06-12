# Fraud-Detection-System

> Mobile-money fraud classifier on PaySim — XGBoost at 99.11% precision and 96.88% recall on a 132K time-based holdout.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB6E2D)](https://xgboost.readthedocs.io/)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.9987-success)](#results)
[![Brier](https://img.shields.io/badge/Brier-0.0005-success)](#calibration)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)

![Project banner — fraud-detection system on PaySim mobile-money data](assets/banner.svg)

## Why?

Mobile-money fraud is a precision problem, not an accuracy problem. The PaySim dataset has a 0.13% positive rate; a model that says "not fraud" every time scores 99.87% accuracy and catches zero fraud. Production fraud-ops workflows freeze customer funds on a flag, so false positives carry direct trust and regulatory cost. This repo builds a classifier honest about that trade-off: precision is held at or above 99% and the model is evaluated on a strict time-based holdout — no future-state leakage.

> **A note on PaySim.** PaySim is widely used in introductory fraud-detection tutorials. The contribution here is not the dataset choice but the evaluation rigour: strict time-based split, calibrated probabilities, cost-sensitive threshold selection, and a five-model comparison on identical feature pipelines. See the [Dataset](#dataset) section for the full methodological rationale.

## Project Structure

```
Fraud-Detection-System/
├── src/
│   ├── train.py          # Training pipeline — data -> model serialisation
│   └── predict.py        # Inference script — load model, score transactions
├── model/
│   └── xgb_fraud_model.pkl   # Serialised trained model (run make train)
├── assets/
│   ├── banner.svg
│   ├── 02_balance_discrepancy_fingerprint.png
│   ├── 03_pr_curve_scoreboard.png
│   ├── 04_calibration_curve.png
│   └── 07_shap_beeswarm.png
├── Fraud Detection System.ipynb   # End-to-end analysis notebook
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
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download PaySim from https://www.kaggle.com/datasets/ealaxi/paysim1
# Place PS_20174392719_1491204439457_log.csv in the project root, then:

make train                          # trains + serialises model to model/
make predict                        # score a single transaction interactively
make predict-csv INPUT=txns.csv     # score a batch CSV
```

## Features

- XGBoost classifier calibrated to a Brier score of 0.0005 (vs. random baseline ~0.0204)
- 99.11% precision / 96.88% recall at the deployment operating point (threshold 0.9989)
- Time-based train/test split at step 490 — no random shuffling, no leakage
- Balance-discrepancy feature engineering carrying ~85% of the predictive signal
- SHAP attribution for per-prediction explanation (computed on a 2,000-row representative sample for runtime efficiency; results are stable across random seeds)
- Five-model comparison (RF, Stacking, XGBoost, LR, LightGBM) on identical features
- Inference script (`src/predict.py`) — scores a single transaction or a full CSV from the command line
- Reproducible pipeline in a single notebook with deterministic seeds

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Core data | `numpy`, `pandas` |
| Modelling | `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn` |
| Explainability | `shap` |
| Serialisation | `joblib` |
| Plotting | `matplotlib`, `seaborn`, `plotly` |
| Environment | `jupyter`, `jupyterlab` |

## Installation

```bash
git clone https://github.com/alvenyuka/Fraud-Detection-System.git
cd Fraud-Detection-System

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the PaySim dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place `PS_20174392719_1491204439457_log.csv` in the project root.

## Usage

### Jupyter Notebook (full analysis)

```bash
jupyter lab "Fraud Detection System.ipynb"
```

### Command-line inference

```bash
make train
make predict
make predict-csv INPUT=my_transactions.csv OUTPUT=scored.csv
```

Example interactive session:

```
-- Transaction details --
  type (TRANSFER / CASH_OUT): TRANSFER
  amount: 950000
  oldbalanceOrg: 950000
  newbalanceOrig: 0
  oldbalanceDest: 0
  newbalanceDest: 0

-- Result --
  Fraud probability : 0.999987
  Operating threshold: 0.9989
  Decision          : FRAUD FLAGGED
```

## Dataset

**PaySim** — a simulator built on real African mobile-money transaction logs, anonymised and time-aligned.

| Property | Value |
|---|---|
| Rows | 6,362,620 |
| Time horizon | 31 days (744 hourly steps) |
| Fraud rate | 0.1291% |
| Active fraud types | `TRANSFER`, `CASH_OUT` only |
| Filtered dataset | 2,770,409 rows (56.5% reduction, zero positive loss) |

## Methodology

### Evaluation strategy

Time-based split at step 490. Random splits leak future state into training and inflate PR-AUC. A production fraud model must predict future transactions from past evidence.

| Split | Steps | Rows | Fraud rate |
|---|---|---|---|
| Train | 1-490 | 2,638,273 | 0.207% |
| Test | 491-743 | 132,136 | 2.084% |

### Why PR-AUC, not accuracy or ROC-AUC

**PR-AUC is the only metric that lives entirely on the precision/recall trade-off within the positive class**, and is the only honest measure of fraud-detection performance under extreme imbalance.

### Operating point selection

The deployment-relevant question is the highest recall achievable while precision stays at or above 99%. For XGBoost, this point sits at a probability threshold of 0.9989.

### Calibration

![Calibration curve — XGBoost predicted probabilities vs. observed fraud rate](assets/04_calibration_curve.png)

### Feature engineering — balance-discrepancy logic

![Balance discrepancy fingerprint — fraud vs. legitimate transactions](assets/02_balance_discrepancy_fingerprint.png)

## Results

### Headline performance

| Metric | Value |
|---|---|
| Model | XGBoost (calibrated) |
| Precision | **99.11%** |
| Recall | **96.88%** |
| F1 | 0.980 |
| PR-AUC | 0.9987 |
| ROC-AUC | 1.0000 * |
| Brier score | 0.0005 |
| Operating threshold | 0.9989 |
| Test window | steps 491-743 (132,136 transactions, 2,754 positives) |

*ROC-AUC of 1.0000 is a documented property of PaySim once balance-discrepancy features are engineered — the accounting identity makes most fraud cases nearly linearly separable. This is an upper bound on dataset learnability. PR-AUC is the primary metric.*

### Model comparison

| Model | PR-AUC | ROC-AUC | Recall @ 99% Precision | Lift vs. `isFlaggedFraud` |
|---|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 35.3x |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 | 35.3x |
| **XGBoost (deployable)** | **0.9987** | **1.0000** | **0.9688** | **35.3x** |
| Logistic Regression | 0.7905 | 0.9796 | 0.4506 | 27.9x |
| LightGBM (default) | 0.2451 | 0.9502 | 0.0000 | 8.7x |
| `isFlaggedFraud` (rule baseline) | 0.0283 | 0.5888 | 0.0000 | 1.0x |

**Notes:** Random Forest / Stacking at PR-AUC = 1.0000 reflects the same PaySim separability property as ROC-AUC above — reported as an empirical upper bound. LightGBM PR-AUC = 0.2451 reflects default hyperparameters with no class-weight correction; under 0.2% positive rate, defaults collapse to predicting only the majority class. A tuned LightGBM with `is_unbalance=True` would be competitive.

![Precision-recall curve scoreboard](assets/03_pr_curve_scoreboard.png)

### Feature attribution

SHAP values computed on a representative 2,000-row sample (stratified by fraud label) for runtime efficiency. Results are stable across samples.

![SHAP beeswarm — per-feature attribution for the deployable XGBoost model](assets/07_shap_beeswarm.png)

## Roadmap

- [x] Time-based evaluation harness
- [x] Five-model comparison with identical pipelines
- [x] Calibration + SHAP attribution
- [x] Cost-sensitive threshold analysis
- [x] Inference script (`src/predict.py`)
- [x] Training pipeline (`src/train.py`)
- [x] Model card (`MODEL_CARD.md`)
- [ ] Streaming inference scaffold (Kafka -> FastAPI)
- [ ] Drift monitoring (PSI on balance-discrepancy features)

## License

MIT — see [`LICENSE`](LICENSE).

## Credits

- **Dataset:** Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.*
- Built by **Alven Yuka** — CPA Finalist (Kenya), Nairobi.

## Connect

[alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · [GitHub](https://github.com/alvenyuka)
