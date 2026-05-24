# Fraud-Detection-System

> Mobile-money fraud classifier on PaySim — XGBoost at 99.11% precision and 96.88% recall on a 132K time-based holdout.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB6E2D)](https://xgboost.readthedocs.io/)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.9987-success)](#results)
[![Brier](https://img.shields.io/badge/Brier-0.0005-success)](#calibration)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)

![Project banner — fraud-detection system on PaySim mobile-money data](banner.svg)

## Why?

Mobile-money fraud is a precision problem, not an accuracy problem. The PaySim dataset has a 0.13% positive rate; a model that says "not fraud" every time scores 99.87% accuracy and catches zero fraud. Production fraud-ops workflows freeze customer funds on a flag, so false positives carry direct trust and regulatory cost. This repo builds a classifier honest about that trade-off: precision is held at or above 99% and the model is evaluated on a strict time-based holdout — no future-state leakage.

## Features

- 📈 **XGBoost classifier** calibrated to a Brier score of 0.0005 (vs. random baseline ~0.0204)
- 🎯 **99.11% precision / 96.88% recall** at the deployment operating point (threshold 0.9989)
- ⏱️ **Time-based train/test split** at step 490 — no random shuffling, no leakage
- 🧮 **Balance-discrepancy feature engineering** carrying ~85% of the predictive signal
- 🔬 **SHAP attribution** for per-prediction explanation
- 📊 **Five-model comparison** (RF, Stacking, XGBoost, LR, LightGBM) on identical features
- ✅ **Reproducible pipeline** in a single notebook with deterministic seeds

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Core data | `numpy`, `pandas` |
| Modelling | `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn` |
| Explainability | `shap` |
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

```bash
jupyter lab "Fraud Detection System.ipynb"
```

The notebook runs end-to-end: data load → time-based split → feature engineering → five-model training → calibration → SHAP attribution. Every figure in this README is regenerated from notebook cells — no hidden state.

## Dataset

**PaySim** — A simulator built on real African mobile-money transaction logs, anonymised and time-aligned. Six transaction types (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`, plus internal `M*` merchant rows). Each transaction carries an origin and destination account, balance snapshots before and after, and an `isFraud` label.

| Property | Value |
|---|---|
| Rows | 6,362,620 |
| Time horizon | 31 days (744 hourly steps) |
| Fraud rate | 0.1291% |
| Active fraud types | `TRANSFER`, `CASH_OUT` only |
| Filtered dataset | 2,770,409 rows (56.5% reduction, zero positive loss) |

## Methodology

### Evaluation strategy

The evaluation uses a **strict time-based split at step 490** (approximately 66% of the simulated horizon). Random splits leak future state into training and inflate PR-AUC, since fraud patterns evolve across the dataset. A production fraud model must predict future transactions from past evidence, and the evaluation must match.

| Split | Steps | Rows | Fraud rate |
|---|---|---|---|
| Train | 1–490 | 2,638,273 | 0.207% |
| Test | 491–743 | 132,136 | 2.084% |

### Why PR-AUC, not accuracy or ROC-AUC

The fraud rate is 0.13%, so a model predicting "not fraud" for every transaction scores 99.87% accuracy and catches zero fraud. ROC-AUC is similarly inflated because the true-negative count dominates the curve regardless of fraud-class performance. **PR-AUC is the only metric that lives entirely on the precision/recall trade-off within the positive class**, and is the only honest measure of fraud-detection performance under extreme imbalance.

### Operating point selection

Fraud-operations workflows freeze customer funds on a flag, so false positives carry direct trust and regulatory cost. The deployment-relevant question is **the highest recall achievable while precision stays at or above 99%**, not the threshold that maximises F1 in a vacuum. For XGBoost, this point sits at a probability threshold of 0.9989.

### Calibration

The deployable model is calibrated against a random-baseline Brier of approximately 0.0204. Calibrated probabilities matter because the downstream fraud-ops team thresholds on score, and uncalibrated tree outputs are not interpretable as probabilities.

![Calibration curve — XGBoost predicted probabilities vs. observed fraud rate](04_calibration_curve.png)

### Feature engineering — balance-discrepancy logic

Raw transaction `amount` does not separate fraud from genuine transactions; both distributions overlap significantly. The discriminative signal lives in the **deviation between expected and actual post-transaction balances**. For a legitimate transfer the accounting identity holds; for a fraudulent one it breaks. Engineering this discrepancy as an explicit feature drives roughly 85% of the model's predictive lift.

![Balance discrepancy fingerprint — fraud vs. legitimate transactions](02_balance_discrepancy_fingerprint.png)

## Results

### Headline performance

| Metric | Value |
|---|---|
| Model | XGBoost (calibrated) |
| Precision | **99.11%** |
| Recall | **96.88%** |
| F1 | 0.980 |
| PR-AUC | 0.9987 |
| ROC-AUC | 1.0000 |
| Brier score | 0.0005 |
| Operating threshold | 0.9989 |
| Test window | steps 491–743 (132,136 transactions, 2,754 positives) |

### Model comparison

All models trained on the same time-based split (steps 1–490 train, 491–743 test) with identical feature pipelines.

| Model | PR-AUC | ROC-AUC | Recall @ 99% Precision | Lift vs. `isFlaggedFraud` |
|---|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 35.3× |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 | 35.3× |
| **XGBoost (deployable)** | **0.9987** | **1.0000** | **0.9688** | **35.3×** |
| Logistic Regression | 0.7905 | 0.9796 | 0.4506 | 27.9× |
| LightGBM (default) | 0.2451 | 0.9502 | 0.0000 | 8.7× |
| `isFlaggedFraud` (rule baseline) | 0.0283 | 0.5888 | 0.0000 | 1.0× |

> **Note on the perfect-score tree models.** Random Forest and the stacking ensemble both score PR-AUC = 1.0000 on the holdout. This is a documented property of the PaySim simulator once the balance-discrepancy features are engineered: the underlying accounting identity makes most fraud cases nearly linearly separable. These results are reported as an upper bound on dataset learnability, not as production candidates. XGBoost at the 99% precision operating point is the model selected for deployment.

![Precision–recall curve scoreboard comparing XGBoost, Random Forest, stacking ensemble, logistic regression and LightGBM on the PaySim holdout](03_pr_curve_scoreboard.png)

### Feature attribution

![SHAP beeswarm — per-feature attribution for the deployable XGBoost model](07_shap_beeswarm.png)

## Roadmap

- [x] Time-based evaluation harness
- [x] Five-model comparison with identical pipelines
- [x] Calibration + SHAP attribution
- [x] Cost-sensitive threshold analysis
- [ ] Streaming inference scaffold (Kafka → FastAPI)
- [ ] Drift monitoring (PSI on balance-discrepancy features)
- [ ] Model card published to Hugging Face

## License

MIT — see [`LICENSE`](LICENSE).

## Credits

- **Dataset:** Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.*
- Built by **Alven Yuka** — CPA Finalist (Kenya), Nairobi.

## Connect

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alvenyuka) · 🐙 [GitHub](https://github.com/alvenyuka)
