"""The headline figures in README.md, checked against the files the code writes.

Each check takes a value from a committed output file, formats it the way the
README writes it (same rounding, same separators), and asserts that exact text
appears in README.md. If a retrain or a re-run changes an output, the matching
README line fails here instead of drifting silently.

Sources (all committed, so no dataset is needed):
- outputs/holdout_metrics.json       holdout PR-AUC, frauds caught, false alarms
- dashboard/data/business_impact.json value stopped by each screen
- dashboard/data/metrics_summary.json walk-forward averages and thresholds
- dashboard/data/feature_importance.json, psi_timeline.csv, scenario_table.json
- outputs/notebook_results.json       data counts from the notebook's run
- model/xgb_fraud_model.pkl           the metrics src/train.py stored with the model
"""
import csv
import json
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
README_RAW = (ROOT / "README.md").read_text(encoding="utf-8")


def collapse_whitespace(text):
    """Turn every run of spaces and line breaks into one space, so wrapped lines still match."""
    return " ".join(text.split())


README_TEXT = collapse_whitespace(README_RAW)


def read_json(relative_path):
    """Load one committed JSON output file."""
    with open(ROOT / relative_path, encoding="utf-8") as handle:
        return json.load(handle)


def as_percent(share, decimals):
    """0.99564 -> '99.6%' (one decimal) or '99.56%' (two decimals)."""
    return f"{share * 100:.{decimals}f}%"


def as_count(number):
    """2754 -> '2,754'."""
    return f"{int(number):,}"


def as_billions(amount, decimals):
    """4190812449.66 -> '4.19 bn'."""
    return f"{amount / 1e9:.{decimals}f} bn"


def as_millions(amount, decimals):
    """46474810.66 -> '46.5 M'."""
    return f"{amount / 1e6:.{decimals}f} M"


def assert_in_readme(expected_text, where):
    """Fail with a readable message when the README does not carry the computed text."""
    assert collapse_whitespace(expected_text) in README_TEXT, (
        f"README should say {expected_text!r} ({where}), but that text is not in README.md"
    )


# ---------------------------------------------------------------------------
# Holdout results (outputs/holdout_metrics.json)
# ---------------------------------------------------------------------------

HOLDOUT = read_json("outputs/holdout_metrics.json")


def test_holdout_counts():
    assert_in_readme(as_count(HOLDOUT["holdout_rows"]) + " TRANSFER and CASH_OUT", "holdout rows")
    assert_in_readme(f"{as_count(HOLDOUT['frauds_caught'])} of the {as_count(HOLDOUT['holdout_frauds'])} frauds",
                     "frauds caught")
    assert_in_readme(f"It froze {HOLDOUT['false_alarms']} honest transfers", "false alarms")


def test_holdout_recall_and_precision():
    caught = HOLDOUT["frauds_caught"]
    recall = caught / HOLDOUT["holdout_frauds"]
    precision = caught / (caught + HOLDOUT["false_alarms"])
    assert_in_readme(f"(recall {as_percent(recall, 1)})", "holdout recall")
    assert_in_readme(f"(precision {as_percent(precision, 1)})", "holdout precision")


def test_holdout_pr_auc_and_threshold():
    assert_in_readme(f"PR-AUC {HOLDOUT['pr_auc']:.4f}", "holdout PR-AUC")
    assert_in_readme(f"| Flag threshold | {HOLDOUT['operating_threshold']:.4f} |", "threshold row")


def test_metrics_stored_with_the_model_match_the_recomputed_file():
    bundle = joblib.load(ROOT / "model" / "xgb_fraud_model.pkl")
    stored = bundle["metrics"]
    assert stored["PR-AUC"] == HOLDOUT["pr_auc"]
    assert stored["Frauds caught"] == HOLDOUT["frauds_caught"]
    assert stored["False alarms"] == HOLDOUT["false_alarms"]
    assert_in_readme(f"| Brier score | {stored['Brier']:.6f} |", "Brier row")
    assert_in_readme(f"kept {bundle['n_estimators']} trees", "tree count")


# ---------------------------------------------------------------------------
# Money on the holdout (dashboard/data/business_impact.json)
# ---------------------------------------------------------------------------

IMPACT = read_json("dashboard/data/business_impact.json")["results"]


def test_value_stopped_by_the_model():
    model = IMPACT["Model, threshold 0.8852"]
    assert_in_readme(f"It stopped {as_percent(model['share_of_fraud_value_stopped'], 2)} of the fraud value",
                     "headline value stopped")
    assert_in_readme(f"**{as_billions(model['fraud_value_stopped'], 3)} "
                     f"({as_percent(model['share_of_fraud_value_stopped'], 2)})**", "model row")
    assert_in_readme(f"| {as_millions(model['fraud_value_missed'], 2)} | {model['honest_transactions_frozen']} |",
                     "model row, missed and frozen")
    assert_in_readme(f"worth {as_millions(model['honest_value_frozen'], 2)} in total", "frozen value")


def test_the_lower_threshold_gives_the_same_result():
    higher = IMPACT["Model, threshold 0.8852"]
    lower = IMPACT["Model, threshold 0.5"]
    assert lower["frauds_caught"] == higher["frauds_caught"]
    assert lower["honest_transactions_frozen"] == higher["honest_transactions_frozen"]


def test_no_screening_and_builtin_flag_rows():
    nothing = IMPACT["No screening"]
    builtin = IMPACT["isFlaggedFraud flag"]
    assert_in_readme(f"| None | 0 | {as_billions(nothing['fraud_value_missed'], 2)} | 0 |", "no-screening row")
    assert_in_readme(f"| {as_millions(builtin['fraud_value_stopped'], 1)} "
                     f"({as_percent(builtin['share_of_fraud_value_stopped'], 1)}) | "
                     f"{as_billions(builtin['fraud_value_missed'], 2)} |", "built-in flag row")
    assert_in_readme(f"The built-in flag caught {builtin['frauds_caught']} of the", "built-in flag count")


def test_precision_at_the_whole_simulation_fraud_rate():
    at_rate = read_json("dashboard/data/business_impact.json")["precision_at_real_fraud_rate"]
    assert_in_readme(f"against {as_percent(at_rate['fraud_rate'], 2)} for TRANSFER and CASH_OUT", "real fraud rate")
    assert_in_readme(f"about {round(at_rate['false_alarms'])} false alarms per million", "false alarms per million")
    assert_in_readme(f"keep {as_percent(at_rate['precision'], 1)}", "precision at real rate")


# ---------------------------------------------------------------------------
# Walk-forward validation (dashboard/data/metrics_summary.json)
# ---------------------------------------------------------------------------

SUMMARY = read_json("dashboard/data/metrics_summary.json")


def test_walk_forward_averages():
    pr_auc = SUMMARY["PR-AUC"]
    assert_in_readme(f"walk-forward PR-AUC {pr_auc['mean']:.4f}", "walk-forward PR-AUC")
    assert_in_readme(f"| {pr_auc['mean']:.4f} ± {pr_auc['std']:.4f} |", "walk-forward table row")
    assert_in_readme(f"| {as_percent(SUMMARY['Precision']['mean'], 1)} / {as_percent(SUMMARY['Recall']['mean'], 1)} |",
                     "walk-forward precision / recall row")


def test_walk_forward_last_window_and_thresholds():
    folds = SUMMARY["folds"]
    last_fold = folds[-1]
    assert_in_readme(f"fell to {as_percent(last_fold['Precision'], 1)} in the last", "last-window precision")
    assert_in_readme(f"moved from {folds[0]['threshold']} to {folds[1]['threshold']}, {folds[2]['threshold']}"
                     f" and {folds[3]['threshold']}", "walk-forward thresholds")
    assert_in_readme(f"({SUMMARY['threshold']['min']} to {SUMMARY['threshold']['max']})", "threshold range")
    assert_in_readme(f"testing hours 351 to 450, 451 to 550, 551 to 650 and 651 to {last_fold['test_end_step']}",
                     "window hours")


# ---------------------------------------------------------------------------
# Drift, explanations and scenario probes
# ---------------------------------------------------------------------------

def test_psi_in_the_last_window():
    last_window = {}
    with open(ROOT / "dashboard" / "data" / "psi_timeline.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["window_start_step"] == "701":
                last_window[row["feature"]] = float(row["psi"])
    assert_in_readme(f"`orig_drain_ratio` reached {last_window['orig_drain_ratio']:.2f}", "drain-ratio PSI")
    assert_in_readme(f"`dest_amount_ratio` {last_window['dest_amount_ratio']:.2f}", "amount-ratio PSI")


def test_shap_shares():
    shares = read_json("dashboard/data/feature_importance.json")
    for entry in shares:
        assert_in_readme(f"`{entry['feature']}`", "SHAP feature name")
        assert_in_readme(f"{entry['importance_pct']:.2f}%", f"SHAP share of {entry['feature']}")
    top_two = shares[0]["importance_pct"] + shares[1]["importance_pct"]
    assert_in_readme(f"carry {round(top_two)}% of the attribution", "top-two SHAP share")


def test_scenario_probes():
    scenarios = read_json("dashboard/data/scenario_table.json")
    full_drain = scenarios["drain_sweep"][-1]
    near_drain = scenarios["drain_sweep"][-2]
    assert full_drain["drain_share"] == 1.0 and near_drain["drain_share"] == 0.99
    assert_in_readme(f"one that moves 100% scores {as_percent(full_drain['fraud_probability'], 1)}", "full drain")
    assert_in_readme(f"a transfer that moves 99% of an account scores {round(near_drain['fraud_probability'] * 100)}%",
                     "99% drain")
    assert_in_readme(f"still scores {scenarios['small_full_drain']['fraud_probability']:.4f}", "small full drain")


# ---------------------------------------------------------------------------
# Data counts (outputs/notebook_results.json)
# ---------------------------------------------------------------------------

NOTEBOOK = read_json("outputs/notebook_results.json")


def test_data_counts():
    data = NOTEBOOK["data"]
    assert_in_readme(f"{as_count(data['transactions'])} transactions", "transaction count in the intro")
    assert_in_readme(f"{as_count(data['frauds'])} are fraud, which is {as_percent(data['fraud_rate'], 2)}",
                     "fraud count")
    assert_in_readme(f"leaves {as_count(data['transfer_and_cash_out'])} rows", "modelled rows")
    pattern = data["balance_patterns_pct"]
    whole_balance = "amount equals the sender's whole balance"
    assert_in_readme(f"in {pattern['fraud'][whole_balance]:.2f}% of frauds", "whole-balance share, fraud")
    assert_in_readme(f"against {pattern['honest'][whole_balance]:.1f}% of legitimate", "whole-balance share, honest")
    assert_in_readme(f"right {as_percent(1 - data['fraud_rate'], 2)} of the time", "flag-nothing accuracy")


def test_split_fraud_rates():
    split = NOTEBOOK["split"]
    assert split["test_rows"] == HOLDOUT["holdout_rows"]
    assert split["test_frauds"] == HOLDOUT["holdout_frauds"]
    text = (f"{as_percent(split['test_fraud_rate'], 2)} against {as_percent(split['train_fraud_rate'], 2)} "
            "in training")
    assert_in_readme(text, "train and holdout fraud rates")


# ---------------------------------------------------------------------------
# The test count quoted under How to run and Project structure
# ---------------------------------------------------------------------------

def test_readme_states_the_collected_test_count(request):
    collected = request.session.items
    files_collected = set()
    for item in collected:
        files_collected.add(Path(str(item.fspath)).name)
    files_on_disk = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        files_on_disk.add(path.name)
    if files_collected != files_on_disk or request.config.getoption("keyword"):
        pytest.skip("only meaningful when the whole test suite is collected")
    count = len(collected)
    assert_in_readme(f"make test        # {count} tests", "How to run test count")
    assert_in_readme(f"tests/          {count} tests", "Project structure test count")
