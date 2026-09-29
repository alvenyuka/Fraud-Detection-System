# Fraud Detection System

> How can a mobile-money operator stop fraudulent transfers without freezing honest customers' money? An XGBoost model on 6.3 million simulated mobile-money transactions: 99.85% precision and 99.56% recall on a time-based holdout, with a live scoring demo.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB6E2D)](https://xgboost.readthedocs.io/)
[![tests](https://github.com/alvenyuka/Fraud-Detection-System/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Fraud-Detection-System/actions/workflows/ci.yml)

![Project banner: fraud-detection system on PaySim mobile-money data](banner.svg)

**Try it:** [live scoring page](https://fraud-detection-alven.vercel.app) · [full dashboard](https://fraud-detection-system-kmeuq7hku8tglnxdpmalfk.streamlit.app/) (free host, allow about 30 seconds to wake)

## The problem

In mobile money, a fraud flag usually freezes the customer's funds. Every false alarm is an honest customer
locked out of their own money, which costs trust and invites regulatory complaints. Every missed fraud is a
direct loss. Fraud is also rare, about 0.13% of transactions here, so a model that never flags anything is
99.87% "accurate" and useless.

The question for a fraud-operations team is therefore not accuracy but precision and recall at a threshold
set by what each kind of mistake costs.

## What I found

| Measure (132,136 transactions after the training period) | Result |
|---|---:|
| Precision, flagged transactions that were fraud | **99.85%** |
| Recall, frauds that were caught | **99.56%** |
| PR-AUC, ranking quality across all thresholds | 0.9993 |
| PR-AUC averaged over 4 later time windows | 0.9986 ± 0.0013 |

- **Fraud in this data has one signature: the sender's account is drained to zero** while the balances do not
  add up. Two engineered features built on that accounting identity carry about three quarters of the model's
  decisions.
- **Testing the live demo exposed a costly error, and I fixed it.** The first model flagged legitimate account
  closures, for example a customer emptying their own $12 account, as certain fraud, because it had learned
  "balance hits zero" on its own. Removing the raw balance inputs stopped that.
- **The threshold has to be revisited.** The cost-optimal cutoff shifts noticeably from one time window to the
  next, so a fixed threshold set once would drift out of date in production.

**What I would recommend to a fraud-operations team:** use a model like this to rank and triage alerts,
choose the threshold from the real cost of a frozen account versus a missed fraud, review it on a schedule,
and add rules for partial-drain fraud, which this model does not catch (see Limitations).

## How it works

1. **Time-based split.** Train on the first 490 simulated hours, test on the rest, so the model never sees
   the future.
2. **Accounting features.** For each transaction, check whether the sender's and recipient's balances move
   by exactly the amount sent. Fraud breaks that identity.
3. **XGBoost with tuned settings**, then probability calibration on a held-out slice of the training period.
4. **Cost-based threshold.** The cutoff is chosen by weighing a missed fraud at 100 times the cost of a false
   alarm, not fixed at 0.5.
5. **Monitoring.** Population Stability Index tracks whether the inputs drift over time, and SHAP explains
   each individual score in the dashboard.

## Run it

```bash
pip install -r requirements.txt
make test      # 28 tests, no dataset needed
# download PaySim from Kaggle into the project root, then:
make train     # train and save the model, a few minutes
make predict   # score a transaction
```

## Limitations

- **PaySim is a simulator, not real payment traffic.** Its fraud becomes almost deterministic once these
  features are built, which is why the scores are so high. They show a sound evaluation method, not the
  accuracy to expect on a real network.
- **It detects full-drain fraud only.** A fraud that skims part of an account without emptying it scores
  close to zero.
- **The test period is fraud-heavy** (2.08% fraud against 0.13% overall), so precision at real-world
  prevalence would be lower.

## More detail

The full write-up, with the walk-forward results and their caveats, the five-model comparison, drift
findings and the tests, is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). The model's intended use and
risks are in [`MODEL_CARD.md`](MODEL_CARD.md).

## License

MIT. See [`LICENSE`](LICENSE). Data: Lopez-Rojas, Elmir and Axelsson (2016), *PaySim: A financial mobile money simulator for fraud detection* ([Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)).

## Connect

Built by Alven Yuka, CPA Finalist and Accounting Specialist at GIZ, Nairobi.

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · 🐙 [GitHub](https://github.com/alvenyuka)
