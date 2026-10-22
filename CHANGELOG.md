# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-22

The first versioned state of the project. Everything below was built before versioning started; the
commit history holds the detail.

### Added

- Time-split XGBoost model on PaySim TRANSFER and CASH_OUT transactions (fit on hours 1 to 392, calibrated on
  393 to 490, scored on 491 to 743), with a threshold picked from a 100:1 missed-fraud to false-alarm cost.
- Holdout result: 2,742 of 2,754 frauds caught, 3 honest transfers frozen, PR-AUC 0.9984, 99.98% of fraud value
  stopped (`outputs/holdout_metrics.json`, `dashboard/data/business_impact.json`).
- Walk-forward validation over four later windows (PR-AUC 0.9971), PSI drift monitoring, SHAP shares and
  scenario probes, all written to `dashboard/data/`.
- Standard-library port of the model (`api/score.py`) behind the live demo, with a parity test against the
  pickle.
- `tests/test_readme_numbers.py`: every headline figure in the README is checked against the committed output
  files, and the README's test count against the collected suite.
- This changelog and a pull request template.

### Fixed

- Two leaks found in an audit: tuning windows that overlapped reporting windows, and walk-forward thresholds
  chosen on the test rows. `tests/test_pipeline_guards.py` guards both.
- The balance-drain shortcut through the raw balance columns, guarded by `tests/test_features.py`.

### Changed

- `.gitignore` now lists `.pytest_cache/`.

[1.0.0]: https://github.com/alvenyuka/Fraud-Detection-System/releases/tag/v1.0.0
