# Fraud Detection on Mobile Money Transactions

A walkthrough of building a fraud classifier on the PaySim dataset: 6,362,620 synthetic transactions, 0.13% fraud rate, 31 days of simulated mobile-money network activity.

The notebook covers exploratory analysis, feature engineering, model selection on a time-based split, and operating-point selection at 99% precision.

**Stack:** Python, pandas, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, matplotlib.
**Notebook:** `Fraud_Detection_System.ipynb`

## Headline result

The deployable model is **XGBoost at 99.11% precision and 96.88% recall** on a 132,136-transaction held-out window (steps 491-743), at a probability threshold of 0.9989. Brier score is 0.0005 against a random-baseline Brier of about 0.0204.

For reference, the `isFlaggedFraud` rule that ships with the dataset catches 0% of fraud at any threshold: it fires only 16 times across 6.3M transactions with no consistent logic behind it. The XGBoost model is about 35x better than that baseline.

### About the perfect-score Random Forest and stacking results

In my evaluation table, Random Forest and the stacking ensemble both score PR-AUC = 1.0000 on the holdout. That is a known property of PaySim once the balance-discrepancy features are engineered: the accounting identity makes most fraud cases nearly linearly separable. I treat those results as an upper bound on what the dataset will allow, not as a deployable model. XGBoost at the 99% precision operating point is what I would actually ship, and that is the model the rest of this README is about.

## Full results table

Held-out future window. Test set: steps 491-743, 132,136 transactions, 2,754 fraud cases.

| Model | PR-AUC | ROC-AUC | Recall @ 99% Precision | Uplift vs baseline |
|---|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 35.3x |
| Stacking Ensemble | 1.0000 | 1.0000 | 1.0000 | 35.3x |
| XGBoost (deployable) | 0.9987 | 1.0000 | 0.9688 | 35.3x |
| Logistic Regression | 0.7905 | 0.9796 | 0.4506 | 27.9x |
| LightGBM | 0.2451 | 0.9502 | 0.0000 | 8.7x |
| `isFlaggedFraud` (rule baseline) | 0.0283 | 0.5888 | 0.0000 | 1.0x |

## Why accuracy is the wrong metric

The fraud rate is 0.13%, so a model that predicts "not fraud" for every transaction scores 99.87% accuracy and catches zero fraud. ROC-AUC is similarly inflated, because the true-negative count dominates the curve. The only honest metric here is PR-AUC, which lives entirely on the precision/recall trade-off within the positive class.

## What the EDA established

A few things mattered for modelling.

Fraud is structurally bounded by transaction type. It occurs only in `TRANSFER` and `CASH_OUT` rows, and never in `CASH_IN`, `DEBIT`, or `PAYMENT`. The fraud pattern is to transfer stolen funds to a mule account and then cash them out. Filtering to the two relevant types reduces the dataset by 56.5% (to 2,770,409 rows) without losing a single positive case.

The shipped fraud flag is near-useless. `isFlaggedFraud` fires 16 times in 6.3M transactions with no defensible threshold behind it. Dropped from the feature set.

Account names carry no signal. Merchant accounts (`M*` prefix) only appear as `PAYMENT` destinations, where fraud is zero. The dataset's anonymisation breaks the mule chain in a way that makes the name fields uninformative on their own. Both `nameOrig` and `nameDest` dropped.

Zero balances are a feature, not a bug. 49% of fraudulent transactions show zero destination balances, vs 0.06% of legitimate ones. I imputed origin zeros as NaN (XGBoost handles missingness natively) and destination zeros with a `-1` sentinel so the fraud signal is preserved.

Fraud does not keep office hours. Off-hours fraud rate (22:00 to 06:00) is around 0.60% against an overall rate of 0.13%. That is largely a denominator effect: legitimate volume collapses overnight while fraud volume stays roughly flat.

## Feature engineering: the balance-discrepancy logic

Raw `amount` does not separate fraud from genuine transactions. The signal lives in the gap between what the balances should be after the transaction and what they actually are.

For a legitimate transfer, the accounting identity holds:

```
sender:   newBalance = oldBalance - amount
receiver: newBalance = oldBalance + amount
```

The engineered features `errorBalanceOrig` and `errorBalanceDest` measure deviation from these identities. They show opposite polarity for fraud vs genuine transactions, which is the discriminative signal the raw features do not carry. Permutation importance attributes around 85% of XGBoost's predictive power to these two features.

## Modelling and validation

Train/test split is time-based at step 490, around 66% of the 31-day window:

| Split | Steps | Rows | Fraud rate |
|---|---|---|---|
| Train | 1-490 | 2,638,273 | 0.207% |
| Test | 491-743 | 132,136 | 2.084% |

I deliberately avoided a random split. Random splits leak future state into training and inflate PR-AUC dishonestly. A production fraud model predicts the future from past data, and the validation has to match.

For class imbalance, I set `scale_pos_weight ≈ 336` on the tree models rather than using SMOTE. XGBoost handles missingness natively, so no imputation pipeline gymnastics are required.

Models evaluated: logistic regression, random forest, LightGBM, XGBoost, a stacking ensemble, and `isFlaggedFraud` as a baseline. All ran through the same harness with the same metrics and the same threshold-selection logic. LightGBM underperformed badly (AP = 0.245); out of the box it is sensitive to extreme imbalance without explicit `scale_pos_weight` tuning. Logistic regression provides a linear floor at AP = 0.79.

## Operating point selection

Fraud ops freezes customer funds on a flag. False positives erode trust and generate regulator complaints, so the deployment question is not "what threshold maximises F1" but "what is the highest recall I can hit while precision stays at or above 99%."

For XGBoost that point sits at a probability threshold of 0.9989, giving precision 99.11% and recall 96.88% on the holdout.
