# Model Card - XGBoost Fraud Classifier (PaySim)

> Following the [Hugging Face model card](https://huggingface.co/docs/hub/model-cards) and [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) standard.

---

## Model Details

| Field | Value |
|---|---|
| **Model type** | XGBoost + isotonic calibration |
| **Version** | 1.2 — tuned hyperparameters, walk-forward validated, drift monitoring added |
| **Date** | 2026 |
| **Author** | Alven Yuka (CPA Finalist, Kenya — awaiting ICPAK membership) |
| **Contact** | alvenyuka2@gmail.com |
| **License** | MIT |
| **Repository** | https://github.com/alvenyuka/Fraud-Detection-System |

### Architecture

- Base: `XGBClassifier`, fit on 80% of the training period. Hyperparameters come from `model/best_params.json` if present (produced by `src/tune.py` — see "Hyperparameter Tuning" below) — otherwise the original hand-picked defaults (500 estimators, max_depth=6, learning_rate=0.1).
- Post-hoc calibration: `CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")`, fit on the held-out 20% of the training period — not on the rows the base model saw, so the calibration curve reflects generalization rather than in-sample fit
- The calibration wrapper ensures output scores are interpretable as probabilities

### Hyperparameter Tuning

The original hyperparameters were hand-picked and never tested against alternatives. `src/tune.py` now searches ~40 combinations (tree depth, learning rate, regularisation, class-imbalance weighting — see the script's own comments for what each setting controls), scored on 3 time-based windows taken entirely from the training period, so the search never touches the data used for final evaluation below.

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

Verified by running `src/train.py` end-to-end against the real PaySim CSV (previous versions of this card reported numbers the script had never actually produced — `cv="prefit"` was removed in scikit-learn ≥1.6, so the script could not run at all until fixed; see [README § Results](README.md#results)).

| Metric | Value | Notes |
|---|---|---|
| Precision | **100.00%** | At operating threshold 0.9989, tuned hyperparameters |
| Recall | **89.47%** | Up from 88.63% before tuning — a modest, honest gain |
| F1 | 0.9444 | At operating threshold 0.9989 |
| PR-AUC | 0.9998 | Primary metric - honest under class imbalance |
| ROC-AUC | 1.0000 * | See caveat below |
| Brier score | 0.000017 | vs. random-baseline ~0.0204; calibrated on a held-out slice, not the training rows |

*PR-AUC and ROC-AUC of 1.0000 are a known property of PaySim once balance-discrepancy features are engineered — treat this as a documented dataset artifact, not evidence of real-world performance. See "Limitations and Risks" below.*

### Walk-forward validation (Step 3, `src/validate.py`)

The table above is one split. `src/validate.py` repeats train → calibrate → test across 4 expanding-window folds spanning the whole dataset, each fold picking its own cost-optimal threshold:

| Metric | Mean (4 folds) | Std dev |
|---|---|---|
| PR-AUC | 0.9997 | ± 0.0004 |
| ROC-AUC | 0.9999 | ± 0.0002 |
| Precision | 0.9954 | ± 0.0065 |
| Recall | 0.9995 | ± 0.0005 |
| F1 | 0.9975 | ± 0.0035 |
| Brier score | 0.0000 | ± 0.0000 |

Low std dev across folds is real evidence this isn't a one-off lucky split. The genuine catch: the per-fold cost-optimal threshold ranges from 0.0078 to 0.9794 — no single fixed threshold is clearly correct across all folds, which is the honest limitation the near-perfect PR-AUC hides. See "Limitations and Risks" below.

---

## Limitations and Risks

- **PaySim is a simulator.** Generalisation to real data is unverified and should be assumed poor without retraining.
- **Drift monitoring is simulated, not real.** `src/monitoring.py` shows what PSI monitoring would look like using PaySim's own time horizon as a stand-in for "time passing in production" — there's no real production traffic behind it yet.
- **Threshold is static per fold.** Each walk-forward fold in `src/validate.py` picks its own cost-optimal threshold; the shipped model still uses one fixed threshold. Different fraud rates require a different operating point.
- False positives freeze customer funds - high precision is a design requirement, not a vanity metric.

---

## Monitoring

`src/monitoring.py` simulates production drift monitoring using Population Stability Index (PSI), since there's no real production traffic to observe. It treats the earliest slice of the PaySim time horizon (steps 1-50) as the "training-time" reference distribution and tracks how far each engineered feature drifts from it in later 50-step windows. Standard PSI thresholds apply: <0.10 stable, 0.10–0.25 moderate shift (worth watching), >0.25 significant shift (investigate).

**Real result, run against the full dataset:** 3 of the 4 engineered features show a significant shift at some point across the time horizon:

| Feature | Worst PSI observed | Verdict |
|---|---|---|
| `dest_balance_discrepancy` | 1.4163 | Significant shift |
| `orig_drain_ratio` | 0.7813 | Significant shift |
| `dest_amount_ratio` | 0.2633 | Significant shift |
| `orig_balance_discrepancy` | 0.1264 | Moderate shift |

This is an honest, useful finding, not a bug to fix: PaySim's transaction volume and fraud mix genuinely change over its 744-step horizon, so a model trained only on the earliest data would need re-calibration (or re-training) as time moves on — exactly the scenario drift monitoring exists to catch. See the dashboard's Monitoring tab for the full timeline chart.

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
@misc{alvenyuka-fraud-2026,
  author  = {Alven Yuka},
  title   = {Fraud Detection System - XGBoost on PaySim},
  year    = {2026},
  url     = {https://github.com/alvenyuka/Fraud-Detection-System}
}
```

Dataset: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.* EMSS Conference.
