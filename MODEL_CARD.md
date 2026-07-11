# Model Card - XGBoost Fraud Classifier (PaySim)

> Following the [Hugging Face model card](https://huggingface.co/docs/hub/model-cards) and [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993) standard.

---

## Model Details

| Field | Value |
|---|---|
| **Model type** | XGBoost + isotonic calibration |
| **Version** | 1.3 — tuned hyperparameters, walk-forward validated, drift monitoring, feature-importance/threshold diagnostics added |
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
"the sender's balance hits zero" as a fraud signal **on its own** — even for
a transaction with a perfectly consistent, zero-discrepancy destination
update. Concretely, a $12 transaction that fully (and correctly) empties a
$12 account scored **100% fraud probability** despite `orig_balance_discrepancy`
and `dest_balance_discrepancy` both being exactly 0 — closing an account or
moving your whole balance somewhere else is completely normal, non-fraudulent
behaviour, but the old model called it certain fraud every time.

The raw balance columns were removed from `FEATURE_COLS`, leaving only
`amount` and the four engineered features above. The model can no longer see
"the balance hit zero" directly — only whether the accounting identity
actually broke — which is the real fraud signal PaySim's discrepancy pattern
is meant to capture.

**This turned out to be a partial fix, not a full one — and re-testing after
the fix caught that.** `orig_drain_ratio` (`amount / oldbalanceOrg`) still
encodes "was the account fully drained" even without the raw balance columns:
a ratio of 1.0 means 100% of the balance moved. Re-running the same test
transaction at different drain fractions makes the remaining effect exact:

| Drain fraction | Fraud probability |
|---|---|
| 10% – 99% | 0.006% (flat) |
| **100%** | **95.3%** |

The probability is flat and near-zero for every fraction up to 99%, then
jumps sharply at exactly 100%. That step function — not a smooth
relationship with how much of the balance moved — is strong evidence that
PaySim's fraud-generation process creates fraud at (almost) exactly 100%
drain, and its legitimate transactions essentially never land on exactly
100%. That is a property of how this simulator generates its labels, not a
bug fixable by dropping more columns — the same correlation would resurface
through `orig_drain_ratio` no matter which raw columns are excluded, because
in this dataset "100% drained" and "fraudulent" really are almost the same
set of rows. A model trained on real transaction data, where legitimate
full-balance transfers and account closures actually occur, would need to be
retrained on that real distribution before this behaviour could be expected
to change.

---

## Evaluation

Verified by running `src/train.py` end-to-end against the real PaySim CSV (previous versions of this card reported numbers the script had never actually produced — `cv="prefit"` was removed in scikit-learn ≥1.6, so the script could not run at all until fixed; see [README § Results](README.md#results)).

| Metric | Value | Notes |
|---|---|---|
| Precision | **99.85%** | At operating threshold 0.4000, picked dynamically by `pick_best_threshold` on the calibration split |
| Recall | **99.56%** | After Step 6's raw-balance-column removal — see § Feature Engineering above |
| F1 | 0.9971 | At operating threshold 0.4000 |
| PR-AUC | 0.9993 | Primary metric - honest under class imbalance |
| ROC-AUC | 0.9998 * | See caveat below |
| Brier score | 0.00017 | vs. random-baseline ~0.0204; calibrated on a held-out slice, not the training rows |

*A near-1.0 PR-AUC/ROC-AUC is a known property of PaySim once balance-discrepancy features are engineered — treat this as a documented dataset artifact, not evidence of real-world performance. See "Limitations and Risks" below.*

### Walk-forward validation (Step 3, `src/validate.py`)

The table above is one split. `src/validate.py` repeats train → calibrate → test across 4 expanding-window folds spanning the whole dataset, each fold picking its own cost-optimal threshold:

| Metric | Mean (4 folds) | Std dev |
|---|---|---|
| PR-AUC | 0.9986 | ± 0.0013 |
| ROC-AUC | 0.9999 | ± 0.0002 |
| Precision | 0.9561 | ± 0.0490 |
| Recall | 0.9998 | ± 0.0004 |
| F1 | 0.9768 | ± 0.0262 |
| Brier score | 0.0002 | ± 0.0001 |

These are the corrected model's numbers, after removing the raw balance columns (see § Feature Engineering above). Precision dropped from 0.9954 and its fold variance grew after the fix — the honest cost of no longer letting the model key off "balance hits zero" as a shortcut; recall improved slightly. Low std dev across folds is still real evidence this isn't a one-off lucky split. The genuine remaining catch: the per-fold cost-optimal threshold still varies a lot across folds — no single fixed threshold is clearly correct across all of them, which is the honest limitation the near-perfect PR-AUC hides. See "Limitations and Risks" below.

---

## Limitations and Risks

- **PaySim is a simulator.** Generalisation to real data is unverified and should be assumed poor without retraining.
- **Drift monitoring is simulated, not real.** `src/monitoring.py` shows what PSI monitoring would look like using PaySim's own time horizon as a stand-in for "time passing in production" — there's no real production traffic behind it yet.
- **Threshold is static per fold.** Each walk-forward fold in `src/validate.py` picks its own cost-optimal threshold; the shipped model still uses one fixed threshold. Different fraud rates require a different operating point.
- **The model barely notices whether the recipient actually received the money.** Pre-deployment scenario testing swept how much of a fully-drained account's balance actually reached the recipient, holding everything else fixed (a $10,000 full-balance TRANSFER, sender drained to zero):

  | % of debited amount credited to recipient | Fraud probability |
  |---|---|
  | 0% (money fully vanishes — classic mule fraud) | 94.77% |
  | 25% / 50% / 75% (partial diversion) | 94.77% (bit-for-bit identical) |
  | 100% (fully consistent, nothing missing) | 76.00% |

  All four "money went missing" cases score *identically* — `dest_balance_discrepancy` only accounts for ~4% of SHAP importance (see `src/explain.py` output), so it barely moves the score even when it's the clearest fraud signal on the page. The flip side of the drain-ratio finding above: this model is a **sender-side full-drain detector**, not a general money-laundering detector. A fraud pattern that partially skims an account *without* fully draining it (e.g. debits 50% of a balance and the recipient gets none of it) scores near **0%** — confirmed directly: a $5,000 partial drain from a $10,000 balance with $0 reaching the recipient scores 0.0061%, indistinguishable from a routine legitimate transaction.

  **A rule-based fix for this was tried and rejected — document this before re-attempting it.** The obvious patch is a safety-net rule layered on top of the ML score: flag any transaction where `dest_balance_discrepancy / amount` is large (the recipient got a lot less than they should have), regardless of what the model says. Tested properly (not just on a convenient sample):

  | Evaluation set | Metric | ML only | ML + shortfall rule |
  |---|---|---|---|
  | Test split (steps 491–743) | Missed fraud / false alarms / cost | 12 / 4 / $12,040 | 10 / 68 / $10,680 |
  | **Train split (steps 1–490)** | Missed fraud / false alarms / cost | 16 / 136 / $17,360 | 16 / **44,996** / **$465,960** |

  The rule looks like a clear win on the test split — 2 more frauds caught, lower cost — but running the *same* rule against the training period (a much larger, more representative sample) causes a false-positive explosion: precision collapses from 97.6% to ~11%, and cost jumps 27×. The root cause: PaySim's destination-balance accounting isn't a strict per-transaction ledger, especially for `CASH_OUT` — the destination is often a shared merchant/agent cash float whose balance legitimately doesn't move 1:1 with any single transaction, for reasons unrelated to fraud. Restricting the rule to `TRANSFER`-only narrowed the damage (4,363 false alarms instead of 46,155) but still nearly quadrupled cost on the training period ($59,630 vs $17,360). **Conclusion: the model's low weighting of `dest_balance_discrepancy` is correct, not a gap to patch — it already learned that this signal isn't reliable at scale, and overriding that with a hand-written rule trades a known, bounded limitation for a much worse, harder-to-predict one.** Closing this blind spot for real would need labeled real-world partial-skim fraud examples to train on, not more feature engineering on this dataset.
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
