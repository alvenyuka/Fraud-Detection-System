# Real-Time Fraud Detection on Mobile Money Transactions

> Production-grade fraud classifier for African mobile money networks. Built around the structural reality of TRANSFER → CASH_OUT mule chains, evaluated on a time-based split, and tuned to a 99% precision operating point that fraud-ops can actually deploy.

**Dataset:** PaySim — 6,362,620 synthetic transactions, **0.1291% fraud rate**, simulated from real African mobile money logs (31 days)
**Stack:** Python · pandas · scikit-learn · XGBoost · LightGBM · imbalanced-learn · SHAP · matplotlib · seaborn · plotly
**Notebook:** [`Fraud_Detection_System.ipynb`](./Fraud_Detection_System.ipynb)

---

## Headline Results

Held-out future window (test set: steps 491–743, 132,136 transactions, 2,754 fraud cases).

| Model | PR-AUC | ROC-AUC | Recall @ 99% Precision | Uplift vs. Baseline |
|---|---:|---:|---:|---:|
| **Random Forest (champion)** | **1.0000** | 1.0000 | 1.0000 | **35.3×** |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 | 35.3× |
| XGBoost | 0.9987 | 1.0000 | 0.9688 | 35.3× |
| Logistic Regression | 0.7905 | 0.9796 | 0.4506 | 27.9× |
| LightGBM | 0.2451 | 0.9502 | 0.0000 | 8.7× |
| `isFlaggedFraud` (shipped baseline) | 0.0283 | 0.5888 | 0.0000 | 1.0× |

- **Operating threshold (XGBoost):** 0.9989 → Precision 99.11%, Recall 96.88%
- **Calibration:** Brier score 0.0005 (random baseline ≈ 0.0204)
- **Uplift:** the champion catches >35× more fraud per unit precision than the rule-based system shipped with the dataset

---

## Why this project exists

Mobile money fraud costs the African financial industry billions annually, and a naive model that predicts *"not fraud"* on every transaction scores 99.87% accuracy while catching zero fraud. That single fact reframes the entire problem:

- **Accuracy is a trap** — the class imbalance does the work for you.
- **ROC-AUC is inflated** — the massive true-negative count drags the curve up regardless of fraud-class performance.
- **PR-AUC is the only honest metric** — it lives entirely on the precision-recall tradeoff among the positive class, which is where the business loss actually sits.

This notebook walks the full reasoning chain — from EDA findings to feature engineering to model selection to operating-point tuning — that a fraud-ops team needs to defend a deployment.

---

## What the EDA established (before any modeling)

1. **Fraud is structurally bounded.** It occurs *only* in `TRANSFER` and `CASH_OUT` transactions — never in `CASH_IN`, `DEBIT`, or `PAYMENT`. The fraud modus operandi is: TRANSFER stolen funds to a mule account, then CASH_OUT. Filtering to those two types reduces the dataset by 56.5% (to 2,770,409 rows) without losing a single positive case.
2. **The shipped fraud flag is near-useless.** `isFlaggedFraud` fires only 16 times across 6.3M transactions — recall of ~0.2%. There is no defensible threshold (amount, balance, repeat-offender) that explains when it fires. **Dropped.**
3. **Account names carry no signal.** Merchant accounts (`M*` prefix) only appear as PAYMENT destinations, where fraud is zero. Fraudulent TRANSFER destinations do not appear as CASH_OUT originators — the expected mule chain is broken by the dataset's anonymization. **Both `nameOrig` and `nameDest` dropped.**
4. **Zero balances are a feature, not a bug.** 49% of fraudulent transactions have zero destination balances vs. 0.06% of genuine ones. Zeros were imputed with **−1 sentinel for destinations** (preserve the fraud signal) and **NaN for origins** (let the model handle missingness natively).
5. **Fraudsters do not keep office hours.** Overall fraud rate is 0.13%; the non-business-hours fraud rate (22:00–06:00) is **0.60%** — nearly 5× higher. Fraud volume is roughly flat across the 24-hour cycle while legitimate volume collapses overnight.

![Dispersion over time](./images/01_dispersion_over_time.jpg)
*Genuine transactions (left) show striped patterns reflecting business hours. Fraud (right) is spread uniformly — the rate spike during off-hours is purely a denominator effect.*

---

## Feature engineering — the balance discrepancy logic

Raw `amount` does not separate fraud from genuine transactions; both have similar distributions. The discriminative signal lives in **what the transaction *should* have looked like vs. what actually happened**.

For any legitimate transaction, this accounting identity should hold:

> Sender:   `newBalance = oldBalance − amount`
> Receiver: `newBalance = oldBalance + amount`

Any deviation is a **balance discrepancy**. The engineered features `errorBalanceOrig` and `errorBalanceDest` show **opposite polarity** for fraud vs. genuine — the literal "fingerprint" the model picks up.

![Balance discrepancy fingerprint](./images/02_balance_discrepancy_fingerprint.png)
*The engineered `errorBalanceDest` feature shows opposite polarity for fraud vs. genuine transactions. This is the discriminative signal raw features alone don't carry.*

---

## Modeling

**Train/test split:** time-based at step 490 (~66% of the 31-day horizon).

| Split | Steps | Rows | Fraud rate |
|---|---|---:|---:|
| Train | 1–490 | 2,638,273 | 0.207% |
| Test | 491–743 | 132,136 | 2.084% |

Random splits leak future information into training and inflate PR-AUC dishonestly — production fraud models always predict the future from the past.

**Class imbalance handling:** `scale_pos_weight = count(genuine) / count(fraud) ≈ 336`. No SMOTE in the champion (see ablation below). XGBoost handles missingness natively — no imputation pipeline gymnastics required.

**Models evaluated:** Logistic Regression, Random Forest, LightGBM, XGBoost, Stacking Ensemble, plus the `isFlaggedFraud` rule as the honest baseline. All share an identical evaluation harness — same train/test split, same metrics, same operating-point logic.

![PR curve scoreboard](./images/03_pr_curve_scoreboard.png)
*Held-out future window. Random Forest and Stacking achieve perfect AP=1.000; XGBoost AP=0.999. LightGBM underperforms badly (AP=0.245) — out-of-the-box LightGBM is sensitive to extreme imbalance without explicit `scale_pos_weight` tuning. Logistic Regression provides the linear floor.*

---

## Operating point — 99% precision

Fraud-ops freezes customer funds when the model flags a transaction. False positives are not just statistical noise — they're eroded customer trust and regulatory complaints. The deployment-relevant question is therefore **the highest recall achievable while precision stays ≥ 99%**, not the threshold that maximizes F1 in a vacuum.

**XGBoost at threshold 0.9989:**

| | Predicted legit | Predicted fraud |
|---|---:|---:|
| **Actually legit** | 129,358 | 24 |
| **Actually fraud** | 86 | 2,668 |

Precision: **99.11%** · Recall: **96.88%** · 24 false positives across 132K transactions.

---

## Robustness — what makes this defensible cold

### Calibration
![Calibration curve](./images/04_calibration_curve.png)
*Brier score = 0.0005 (vs. ~0.0204 for a base-rate model). Predicted probabilities track observed fraud frequency tightly — fraud-ops can rank alerts by score and trust the ordering.*

### Permutation importance
![Permutation importance](./images/05_permutation_importance.png)
*Tree-based importance over-weights correlated features. Permutation importance is the ground-truth check: shuffle each feature on the test set, measure the PR-AUC drop. The engineered `errorBalanceOrig` and `newBalanceOrig` features carry the irreplaceable signal — `errorBalanceOrig` alone accounts for ~85% of the model's predictive power.*

### Cost-sensitive threshold sweep
![Cost-sensitive threshold](./images/06_cost_sensitive_threshold.png)
*The 99%-precision threshold (0.999) is conservative. Under a 10:1 cost ratio (FN:FP), the cost-minimising threshold drops to 0.300. This sweep gives the client the data to pick a threshold that matches their actual cost structure, not an arbitrary statistical default.*

### SMOTE ablation
Standard textbook advice says "use SMOTE for imbalanced classification." For *time-series* fraud, this is often wrong — synthetic minority rows interpolate across temporal regimes the future does not sample from. The ablation:

| Config | PR-AUC | Δ |
|---|---:|---:|
| XGBoost + `scale_pos_weight` | 0.9987 | — |
| XGBoost + SMOTE | 0.9992 | +0.0005 |

The delta is negligible. `scale_pos_weight` remains the conservative default — adding a resampling step that could break under temporal drift in production isn't justified by 5 basis points of PR-AUC.

---

## Explainability — SHAP

![SHAP beeswarm](./images/07_shap_beeswarm.png)
*TreeExplainer on 2,000 held-out test rows. `errorBalanceOrig` dominates global attribution — the feature engineered from the accounting identity, not any raw column. The model also surfaces top-5 highest-risk predictions with their top-3 SHAP drivers each, in the format fraud analysts read directly: alert + reason codes + magnitudes.*

---

## Limitations (and what would change in production)

Stated explicitly because the gap between a notebook and a deployed system is where most portfolio projects quietly break:

1. **No agent-tier features.** East African mobile-money systems have a registered retail agent layer that handles cash-in/cash-out. Agents carry identifiable fraud signatures (velocity spikes, geographic anomalies, concentration in known compromised SIMs). PaySim collapses the agent tier into the `CASH_OUT` type.
2. **No account-age features.** New accounts carry disproportionate fraud risk on real books. PaySim has no account-creation timestamps; the signal is unrecoverable from the available columns.
3. **No graph features.** Every transaction is scored in isolation. Mule networks — A → B → C → D within 24 hours — are a multi-hop pattern that single-transaction scoring misses by construction.
4. **Concept drift posture, not a drift system.** A calibration curve and walk-forward evaluation establish posture; a deployed system also needs a Population Stability Index (PSI) monitor and distribution-drift alarms. That's infrastructure, not notebook work.
5. **Single holdout evaluation of the final threshold.** The 99%-precision threshold is selected on the held-out future window — methodologically defensible, but a production system would re-tune on rolling windows.

---

## How to run

```bash
git clone git@github.com:alvenyuka/Fraud-Detection-System.git
cd Fraud-Detection-System
pip install -r requirements.txt
jupyter lab Fraud_Detection_System.ipynb
```

The PaySim CSV is **not** included in the repo (see `.gitignore`). Download it from Kaggle:
**Synthetic Financial Datasets For Fraud Detection** → `PS_20174392719_1491204439457_log.csv`
Place it in the repo root and update the `data_path` entry in the `CONFIG` dictionary at the top of the notebook.

---

## About

Built by **Alven Yuka** — Financial Data Scientist, Nairobi.
CPA Finalist (Strathmore) · BSc Finance (Co-operative University of Kenya) · ALX Africa Data Science · 3 years GIZ accounting operations.

Portfolio focus: **African fintech and DFI risk modeling** — fraud detection, credit scoring, and financial model replication, with the finance-first foundation that makes the modeling defensible to a credit committee, not just a data science panel.

🔗 [GitHub](https://github.com/alvenyuka) · [LinkedIn](https://www.linkedin.com/in/alvenyuka)
