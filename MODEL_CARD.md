# Model Card - XGBoost Fraud Classifier (PaySim)

> Following the [Hugging Face model card](https://huggingface.co/docs/hub/model-cards) and [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) standard.

---

## Model Details

| Field | Value |
|---|---|
| **Model type** | XGBoost + isotonic calibration |
| **Version** | 1.5: re-tuned only on windows that precede every reporting split; walk-forward thresholds chosen on calibration data; drift monitoring, SHAP diagnostics, live scoring demo |
| **Date** | 2026 |
| **Author** | Alven Yuka (CPA Finalist) |
| **Contact** | alvenyuka2@gmail.com |
| **License** | MIT |
| **Repository** | https://github.com/alvenyuka/Fraud-Detection-System |

### Architecture

- Base: `XGBClassifier`, fit on 80% of the training period. Hyperparameters come from `model/best_params.json` if present (produced by `src/tune.py`, see "Hyperparameter Tuning" below), otherwise the original hand-picked defaults (500 estimators, max_depth=6, learning_rate=0.1).
- Post-hoc calibration: `CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")`, fit on the held-out 20% of the training period, not on the rows the base model saw, so the calibration curve reflects generalization rather than in-sample fit
- The calibration wrapper ensures output scores are interpretable as probabilities

### Hyperparameter Tuning

`src/tune.py` searches 40 combinations with Optuna (tree depth, learning rate, regularisation, row and column
sampling, class weighting). Each combination is scored by average PR-AUC over three time windows that all end
by step 350 (train up to step 200, 250 or 300; score the next 50 steps), so no row used to choose the settings
is used to report a result: walk-forward testing starts at step 351 and the holdout at step 491. Early stopping
inside each window watches the last fifth of that window's training steps, never the rows being scored.

The selected settings (`model/best_params.json`): 500 trees, max_depth 4, learning_rate 0.20,
subsample 0.60, colsample_bytree 0.81, min_child_weight 8, gamma 1.67, scale_pos_weight 2.14.

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

**Time-based split at step 490.** No random shuffling. All training data precedes all test data.

| Split | Steps | Rows |
|---|---|---|
| Train | 1-490 | 2,638,273 |
| Test | 491-743 | 132,136 |

---

## Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `amount` | raw | Transaction size |
| `orig_balance_discrepancy` | `oldbalanceOrg - amount - newbalanceOrig` | Should be ~0 for clean transactions |
| `dest_balance_discrepancy` | `oldbalanceDest + amount - newbalanceDest` | Should be ~0 for clean transactions |
| `orig_drain_ratio` | `amount / (oldbalanceOrg + eps)` | Proportion of origin balance drained |
| `dest_amount_ratio` | `amount / (oldbalanceDest + eps)` | Amount relative to destination balance |

### Why the raw balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) are *not* model inputs

An earlier version of this model fed the raw balance columns into the model
alongside the engineered features above. Manual testing of the live dashboard
found this let the model take a shortcut: PaySim's simulated fraud almost
always drains the sender's account to exactly zero, so the model learned
"the sender's balance hits zero" as a fraud signal **on its own**, even for
a transaction with a perfectly consistent, zero-discrepancy destination
update. Concretely, a $12 transaction that fully (and correctly) empties a
$12 account scored **100% fraud probability** despite `orig_balance_discrepancy`
and `dest_balance_discrepancy` both being exactly 0. Closing an account or
moving your whole balance somewhere else is completely normal, non-fraudulent
behaviour, but the old model called it certain fraud every time.

The raw balance columns were removed from `FEATURE_COLS`, leaving only
`amount` and the four engineered features above. The model can no longer see
"the balance hit zero" directly, only whether the accounting identity
actually broke, which is the real fraud signal PaySim's discrepancy pattern
is meant to capture.

**This turned out to be a partial fix, not a full one, and re-testing after
the fix caught that.** `orig_drain_ratio` (`amount / oldbalanceOrg`) still
encodes "was the account fully drained" even without the raw balance columns:
a ratio of 1.0 means 100% of the balance moved. Re-running the same test
transaction at different drain fractions makes the remaining effect exact:

| Drain fraction | Fraud probability |
|---|---|
| 10% - 99% | 0.00% to 0.01% |
| **100%** | **85.0%** |

The probability stays near zero for every fraction up to 99%, then
jumps sharply at exactly 100%. That step function, not a smooth
relationship with how much of the balance moved, is strong evidence that
PaySim's fraud-generation process creates fraud at (almost) exactly 100%
drain, and its legitimate transactions essentially never land on exactly
100%. That is a property of how this simulator generates its labels, not a
bug fixable by dropping more columns: the same correlation would resurface
through `orig_drain_ratio` no matter which raw columns are excluded, because
in this dataset "100% drained" and "fraudulent" really are almost the same
set of rows. A model trained on real transaction data, where legitimate
full-balance transfers and account closures actually occur, would need to be
retrained on that real distribution before this behaviour could be expected
to change.

---

## Evaluation

Produced by running `src/train.py` end to end against the PaySim CSV. Test set: the 132,136 transactions after
step 490, 2.08% of them fraud.

| Metric | Value | Notes |
|---|---|---|
| Precision | **98.42%** | At operating threshold 0.0123, chosen by `pick_best_threshold` on the calibration split |
| Recall | **99.60%** | |
| F1 | 0.9901 | |
| PR-AUC | 0.9996 | Primary metric, the one that stays informative under class imbalance |
| ROC-AUC | 1.0000 (rounded) * | See caveat below |
| Brier score | 0.000138 | Calibrated on a held-out slice of the training period, not the training rows |

The threshold is low because the calibrated probabilities are honest about rarity (0.21% fraud in training)
and a missed fraud is costed at 100 times a false alarm, so the cost-minimising cutoff sits well below 0.5.

*A near-1.0 PR-AUC/ROC-AUC is a known property of PaySim once balance-discrepancy features are engineered;
treat this as a documented dataset artifact, not evidence of real-world performance. See "Limitations and
Risks" below.*

### Walk-forward validation (`src/validate.py`)

The table above is one split. `src/validate.py` repeats train, calibrate and test across 4 expanding-window
folds testing steps 351-450, 451-550, 551-650 and 651-743. Each fold chooses its own cost-optimal threshold on
its calibration split, never on the test rows, and none of the folds overlaps a tuning window.

| Metric | Mean (4 folds) | Std dev |
|---|---|---|
| PR-AUC | 0.9983 | ± 0.0019 |
| ROC-AUC | 0.9998 | ± 0.0002 |
| Precision | 0.9898 | ± 0.0083 |
| Recall | 0.9973 | ± 0.0022 |
| F1 | 0.9936 | ± 0.0037 |
| Brier score | 0.0001 | ± 0.0001 |

Per-fold thresholds were 0.50, 0.89, 0.019 and 0.017. The cost-optimal cutoff moves a long way from one
period to the next, so no single fixed threshold is clearly right across all of them. See "Limitations and
Risks" below.

**Earlier versions of this card** reported walk-forward precision and recall from a version of
`src/validate.py` that chose each fold's threshold on that fold's own test labels (three folds showed recall
of exactly 1.0), and hyperparameters tuned on windows that overlapped two folds and the holdout. Both are
fixed; the figures above replace them, and the holdout precision moved from 99.85% to 98.42% as a result.

---

## Limitations and Risks

- **PaySim is a simulator.** Generalisation to real data is unverified and should be assumed poor without retraining.
- **Drift monitoring is simulated, not real.** `src/monitoring.py` shows what PSI monitoring would look like using PaySim's own time horizon as a stand-in for "time passing in production"; there's no real production traffic behind it yet.
- **Threshold is static per fold.** Each walk-forward fold in `src/validate.py` picks its own cost-optimal threshold; the shipped model still uses one fixed threshold. Different fraud rates require a different operating point.
- **Thread count is part of the model.** XGBoost draws its row-subsample mask from per-thread RNG streams, so a fixed `random_state` alone does not make a fit with `subsample` below 1.0 thread-invariant (measured on a 60,000-row synthetic frame: up to 0.093 apart in predicted probability between one thread and four at `subsample=0.84`). `n_jobs` is therefore fixed at 4 in `train.py`, `validate.py` and `tune.py`, so a re-run reproduces these figures; changing it will move them slightly.
- **The model barely notices whether the recipient actually received the money.** Pre-deployment scenario testing swept how much of a fully-drained account's balance actually reached the recipient, holding everything else fixed: a $10,000 full-balance TRANSFER, sender drained to zero, **recipient holding $2,000 beforehand**. That opening recipient balance is part of the scenario, not a detail, because the score moves with it; a table that omits it cannot be reproduced.

  | % of debited amount credited to recipient | Fraud probability |
  |---|---|
  | 0% (money fully vanishes, classic mule fraud) | 95.73% |
  | 25% / 50% / 75% (partial diversion) | 95.73% (identical) |
  | 100% (fully consistent, nothing missing) | 85.00% |

  `src/scenarios.py` regenerates this from the committed model and writes `dashboard/data/scenario_table.json`. It needs no dataset: `make scenarios`.

  All four "money went missing" cases score *identically*: `dest_balance_discrepancy` accounts for only 6% of mean absolute SHAP value (see `src/explain.py` output), so it barely moves the score even when it's the clearest fraud signal on the page. The flip side of the drain-ratio finding above: this model is a **sender-side full-drain detector**, not a general money-laundering detector. A fraud pattern that partially skims an account *without* fully draining it (e.g. debits 50% of a balance and the recipient gets none of it) scores near **0%**, confirmed directly and reproduced by the same script: a $5,000 partial drain from a $10,000 balance with $0 reaching the recipient scores 0.0071%, indistinguishable from a routine legitimate transaction.

  **A rule-based fix for this was tried and rejected; document this before re-attempting it.** (Measured on an earlier model version; the conclusion does not depend on the model's exact scores.) The obvious patch is a safety-net rule layered on top of the ML score: flag any transaction where `dest_balance_discrepancy / amount` is large (the recipient got a lot less than they should have), regardless of what the model says. Tested properly (not just on a convenient sample):

  | Evaluation set | Metric | ML only | ML + shortfall rule |
  |---|---|---|---|
  | Test split (steps 491-743) | Missed fraud / false alarms / cost | 12 / 4 / $12,040 | 10 / 68 / $10,680 |
  | **Train split (steps 1-490)** | Missed fraud / false alarms / cost | 16 / 136 / $17,360 | 16 / **44,996** / **$465,960** |

  The rule looks like a clear win on the test split: 2 more frauds caught, lower cost, but running the *same* rule against the training period (a much larger, more representative sample) causes a false-positive explosion: precision collapses from 97.6% to ~11%, and cost jumps 27x. The root cause: PaySim's destination-balance accounting isn't a strict per-transaction ledger, especially for `CASH_OUT`; the destination is often a shared merchant/agent cash float whose balance legitimately doesn't move 1:1 with any single transaction, for reasons unrelated to fraud. Restricting the rule to `TRANSFER`-only narrowed the damage (4,363 false alarms instead of 46,155) but still nearly quadrupled cost on the training period ($59,630 vs $17,360). **Conclusion: the model's low weighting of `dest_balance_discrepancy` is correct, not a gap to patch. It already learned that this signal isn't reliable at scale, and overriding that with a hand-written rule trades a known, bounded limitation for a much worse, harder-to-predict one.** Closing this blind spot for real would need labeled real-world partial-skim fraud examples to train on, not more feature engineering on this dataset.
- False positives freeze customer funds. High precision is a design requirement, not a vanity metric.

---

## Monitoring

`src/monitoring.py` simulates production drift monitoring using Population Stability Index (PSI), since there's no real production traffic to observe. It treats the earliest slice of the PaySim time horizon (steps 1-50) as the "training-time" reference distribution and tracks how far each engineered feature drifts from it in later 50-step windows. Standard PSI thresholds apply: <0.10 stable, 0.10-0.25 moderate shift (worth watching), >0.25 significant shift (investigate).

**Real result, run against the full dataset:** 3 of the 4 engineered features show a significant shift at some point across the time horizon:

| Feature | Worst PSI observed | Verdict |
|---|---|---|
| `dest_balance_discrepancy` | 1.4163 | Significant shift |
| `orig_drain_ratio` | 0.7813 | Significant shift |
| `dest_amount_ratio` | 0.2633 | Significant shift |
| `orig_balance_discrepancy` | 0.1264 | Moderate shift |

This is a useful finding, not a bug to fix: PaySim's transaction volume and fraud mix genuinely change over its 744-step horizon, so a model trained only on the earliest data would need re-calibration (or re-training) as time moves on, exactly the scenario drift monitoring exists to catch. See the dashboard's Monitoring tab for the full timeline chart.

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
