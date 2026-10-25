# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A change to any figure
in `outputs/`, `dashboard/data/` or the model is a new minor version at least. A
version's date is the day its code was finished; its tag is created when the pull
request that carries it is merged.

## [Unreleased]

## [1.0.0] - 2026-10-25

The first versioned state of the project. Everything below was built before versioning started; the
commit history holds the detail.

### Added

- Time-split XGBoost model on PaySim TRANSFER and CASH_OUT transactions (fit on hours 1 to 392, calibrated on
  393 to 490, scored on 491 to 743), with a threshold picked on the calibration hours from two assumed costs:
  1,000 units per missed fraud and 10 per false alarm (`src/features.py`).
- Holdout result: 2,742 of 2,754 frauds caught, 3 honest transfers frozen, PR-AUC 0.9984, 99.98% of fraud value
  stopped (`outputs/holdout_metrics.json`, `dashboard/data/business_impact.json`).
- Walk-forward validation over four later windows (PR-AUC 0.9971), PSI drift monitoring, SHAP shares and
  scenario probes, all written to `dashboard/data/`.
- Standard-library port of the model (`api/score.py`) behind the live demo, with a parity test against the
  pickle.
- `tests/test_readme_numbers.py`: every number in the README's prose, tables and image descriptions is read in
  reading order and compared with an expected list. Each expected number is formatted from a committed output
  file (`outputs/holdout_metrics.json`, `dashboard/data/*`, `outputs/notebook_results.json`, the metrics stored
  in the model file), read from a setting in `src/`, tied to a committed notebook chart, found in
  `MODEL_CARD.md` as history, or listed as a fixed wording number with its reason. A mistyped number anywhere
  fails. The README's test count must match `tests/`.
- This changelog and a pull request template.

### Fixed

- Two leaks found in an audit: tuning windows that overlapped reporting windows, and walk-forward thresholds
  chosen on the test rows. `tests/test_pipeline_guards.py` guards both.
- The balance-drain shortcut through the raw balance columns, guarded by `tests/test_features.py`.
- README: Results opens by saying every figure comes from a simulator, so performance on a real operator's
  transfers is unproven; PR-AUC is listed first; the precision bullet gives the 99.2% expected at the whole
  simulation's lower fraud rate; the threshold step names both assumed costs and where they are set; the flat
  result from 0.5 to 0.8852 cites `dashboard/data/threshold_cost_curve.csv`; Limits no longer says the method
  "carries over" to real data; the balance-fingerprint caption reads the chart the right way round (honest
  senders' unrecorded balances drive the 0.99; fraud shows on the recipient side, 0.65).
- Demo Limits card: precision at the simulator's fraud rate uses the TRANSFER and CASH_OUT rate (0.30%, about 23
  false alarms per million, 99.2% precision, from `dashboard/data/business_impact.json`) instead of the
  all-types 0.13%. Text only; the live page changes after a redeploy.

### Changed

- `.gitignore` now lists `.pytest_cache/`.

[Unreleased]: https://github.com/alvenyuka/Fraud-Detection-System/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/alvenyuka/Fraud-Detection-System/releases/tag/v1.0.0
