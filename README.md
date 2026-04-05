[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](.)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.674-brightgreen)]()
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-7B2D8B)]()
[![CI](https://github.com/alvenyuka/Fraud-Detection-System/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Fraud-Detection-System/actions)
[![Status](https://img.shields.io/badge/Status-Portfolio%20Complete-success)]()

# Real-Time Fraud Detection System for Financial Transactions

## Project Overview

Financial fraud costs the global economy over $30 billion annually. Mobile money platforms and digital payment systems are particularly vulnerable to sophisticated fraud schemes including account takeover, synthetic identity fraud, and transaction laundering.

This project builds an end-to-end fraud detection pipeline for a digital payments platform, simulating the data science workflow at companies like Wave, Stripe, or PayPal. The system must:
- Detect fraudulent transactions in near real-time
- Minimize false positives to avoid blocking legitimate customers
- Handle extreme class imbalance (fraud < 1% of transactions)
- Provide explainable decisions for compliance and audit

## Key Stakeholders & Business Impact

- **Risk & Compliance Team:** Needs interpretable fraud scores for regulatory reporting
- **Operations Team:** Requires actionable alerts with low false-positive rates
- **Product Team:** Needs seamless customer experience (minimal friction for legitimate users)
- **Executive Leadership:** Wants measurable fraud loss reduction metrics

### Business Impact Metrics
- **Primary:** Fraud detection rate (recall) > 85%
- **Secondary:** False positive rate < 5% of flagged transactions
- **Financial:** Estimated $2.5M annual fraud loss reduction
- **Operational:** 60% reduction in manual review queue

---

## Dataset

A realistic synthetic financial transaction dataset was generated to mirror real-world mobile money / digital payment patterns, incorporating:
- Temporal patterns (time-of-day, day-of-week effects)
- Customer behavioral profiles
- Geographic features
- Transaction velocity and amount patterns
- Realistic fraud patterns (account takeover, synthetic identity, etc.)

**Dataset characteristics:**
- 200,000 transactions
- 1.8% fraud rate (3,599 fraudulent transactions)
- Date range: 2023-01-01 to 2024-02-22
- 15 raw features, expanded to 31 engineered features

---

## Feature Engineering

Advanced features engineered to capture fraud indicators:

| Category | Features |
|----------|----------|
| **Amount deviation** | amount_to_avg_ratio, log_amount, amount_pctile_by_type, amount_zscore, is_amount_outlier |
| **Velocity** | txn_count_last_5, amount_sum_last_5, amount_mean_last_5, amount_velocity_ratio |
| **Behavioral** | type_diversity, channel_diversity, is_unusual_channel |
| **Risk composite** | risk_composite (combines night, outlier, new account, unusual channel) |
| **Time-based** | hour, day_of_week, is_weekend, is_night |
| **Customer profile** | customer_age_days, customer_avg_txn |

---

## Model Development

### Models Evaluated

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Train Time (s) |
|-------|---------|--------|----|-----------|--------|----------------|
| Logistic Regression | 0.9361 | 0.6740 | 0.1432 | 0.0784 | 0.8232 | 10.21 |
| Random Forest | 0.9373 | 0.6643 | 0.5669 | 0.5175 | 0.6266 | 27.01 |
| XGBoost | 0.9371 | 0.6641 | 0.4392 | 0.3341 | 0.6407 | 5.81 |
| LightGBM | 0.9379 | 0.6662 | 0.7535 | 0.9863 | 0.6096 | 4.37 |
