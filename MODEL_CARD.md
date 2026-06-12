# Model Card - XGBoost Fraud Classifier (PaySim)

> Following the [Hugging Face model card](https://huggingface.co/docs/hub/model-cards) and [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) standard.

---

## Model Details

| Field | Value |
|---|---|
| **Model type** | XGBoost + isotonic calibration |
| **Version** | 1.0 |
| **Date** | 2024 |
| **Author** | Alven Yuka (CPA Finalist, Kenya) |
| **Contact** | alvenyuka2@gmail.com |
| **License** | MIT |
| **Repository** | https://github.com/alvenyuka/Fraud-Detection-System |

### Architecture

- Base: `XGBClassifier` (500 estimators, max_depth=6, learning_rate=0.1)
- Post-hoc calibration: `CalibratedClassifierCV(method="isotonic", cv="prefit")`
- The calibration wrapper ensures output scores are interpretable as probabilities

---

## Intended Use

### Primary use case

Flagging fraudulent `TRANSFER` and `CASH_OUT` transactions in mobile-money systems similar to the PaySim simulation.

### Intended users

- Risk analytics teams building fraud alerting systems
- Data scientists benchmarking fraud detection approaches
- Recruiters and collaborators evaluating applied ML work

### Out-of-scope uses

- **Real production deployment without retraining.** PaySim is a simulator; a production system must be retrained on real transaction data with proper data governance.
- **Transaction types other than TRANSFER and CASH_OUT.** Fraud in PaySim occurs exclusively in these two types; the model is undefined on `CASH_IN`, `DEBIT`, `PAYMENT`.
- **Financial decisions without human review.** The model flags; a human or rules engine should adjudicate.

---

## Training Data

**PaySim** - a synthetic mobile-money transaction dataset built from anonymised real-world logs of a service operating in Africa.

| Property | Value |
|---|---|
| Source | [Kaggle - PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| Total rows | 6,362,620 |
| Filtered rows (TRANSFER + CASH_OUT) | 2,770,409 |
| Time horizon | 744 steps (31 simulated days) |
| Fraud rate (filtered) | 0.2069% (train) / 2.084% (test) |

**Time-based split at step 490** - no random shuffling. All training data precedes all test data.

| Split | Steps | Rows |
|---|---|---|
| Train | 1-490 | 2,638,273 |
| Test | 491-743 | 132,136 |

---

## Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `orig_balance_discrepancy` | `oldbalanceOrg - amount - newbalanceOrig` | Should be ~0 for clean transactions |
| `dest_balance_discrepancy` | `oldbalanceDest + amount - newbalanceDest` | Should be ~0 for clean transactions |
| `orig_drain_ratio` | `amount / (oldbalanceOrg + eps)` | Proportion of origin balance drained |
| `dest_amount_ratio` | `amount / (oldbalanceDest + eps)` | Amount relative to destination balance |

SHAP attribution confirms these four features plus raw balance fields account for ~85% of the model's predictions.

---

## Evaluation

| Metric | Value | Notes |
|---|---|---|
| Precision | **99.11%** | At operating threshold 0.9989 |
| Recall | **96.88%** | At operating threshold 0.9989 |
| F1 | 0.980 | At operating threshold 0.9989 |
| PR-AUC | 0.9987 | Primary metric - honest under class imbalance |
| ROC-AUC | 1.0000 * | See caveat below |
| Brier score | 0.0005 | vs. random-baseline ~0.0204 |

*ROC-AUC of 1.0000 is a known property of PaySim once balance-discrepancy features are engineered. PR-AUC is the primary metric.*

---

## Limitations and Risks

- **PaySim is a simulator.** Generalisation to real data is unverified and should be assumed poor without retraining.
- **No drift detection.** In production, PSI monitoring on balance-discrepancy features is recommended.
- **Threshold is static.** The 0.9989 threshold was calibrated for the PaySim holdout. Different fraud rates require a different operating point.
- False positives freeze customer funds - high precision is a design requirement, not a vanity metric.

---

## How to Use

```bash
# Train (requires PaySim CSV)
make train DATA=PS_20174392719_1491204439457_log.csv

# Score a single transaction interactively
make predict

# Score a batch CSV
make predict-csv INPUT=transactions.csv OUTPUT=scored.csv
```

See [`src/predict.py`](src/predict.py) and [`src/train.py`](src/train.py).

---

## Citation

```
@misc{alvenyuka-fraud-2024,
  author  = {Alven Yuka},
  title   = {Fraud Detection System - XGBoost on PaySim},
  year    = {2024},
  url     = {https://github.com/alvenyuka/Fraud-Detection-System}
}
```

Dataset: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.* EMSS Conference.
