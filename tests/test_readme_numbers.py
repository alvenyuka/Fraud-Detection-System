"""Every number in README.md, in reading order, checked against where it comes from.

The README is typed by hand. This file reads every number in its prose, tables
and image descriptions, one occurrence at a time, in the order they appear, and
compares that sequence with an expected list. Each expected number is one of
five kinds:

- a pipeline result, formatted at the README's precision from a file the code
  writes: outputs/holdout_metrics.json, dashboard/data/business_impact.json,
  dashboard/data/metrics_summary.json, the other dashboard/data files, the
  metrics src/train.py stores in model/xgb_fraud_model.pkl, or
  outputs/notebook_results.json;
- a setting the code runs with, read from src/ (the split hour, the windows,
  the two costs behind the threshold);
- a figure the executed notebook draws, listed with the figure file;
- a figure from the project's history, found in MODEL_CARD.md;
- a fixed number that is part of the wording (a step number, "1.0 is perfect",
  a citation year), listed with the reason it is fixed.

Code blocks, inline code and link targets are skipped; the test count in the
How to run block has its own test at the end. A re-run that moves a result, or
an edit that mistypes any number anywhere, changes the sequence and fails here.
Adding or rewording a number in the README means adding it here too, with its
source.

No data is needed: every source file is committed, so this runs in CI.
"""
import ast
import csv
import json
import re
from pathlib import Path

import joblib

import config
import features

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def read_json(relative_path):
    # Load one committed JSON output file
    with open(ROOT / relative_path, encoding="utf-8") as handle:
        return json.load(handle)


HOLDOUT = read_json("outputs/holdout_metrics.json")
IMPACT_FILE = read_json("dashboard/data/business_impact.json")
IMPACT = IMPACT_FILE["results"]
AT_REAL_RATE = IMPACT_FILE["precision_at_real_fraud_rate"]
SUMMARY = read_json("dashboard/data/metrics_summary.json")
SHARES = read_json("dashboard/data/feature_importance.json")
SCENARIOS = read_json("dashboard/data/scenario_table.json")
NOTEBOOK_RESULTS = read_json("outputs/notebook_results.json")
BUNDLE = joblib.load(ROOT / "model" / "xgb_fraud_model.pkl")
MODEL_CARD = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")


def psi_in_last_window():
    # PSI of each feature in the last monitoring window (hours 701 to 743)
    last_window = {}
    with open(ROOT / "dashboard" / "data" / "psi_timeline.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["window_start_step"] == "701":
                last_window[row["feature"]] = float(row["psi"])
    return last_window


def count_collected_tests():
    # Count the tests pytest collects: one per test function, or one per value
    # of a literal @pytest.mark.parametrize list
    total = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
                continue
            cases = 1
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", "") == "parametrize":
                    cases = cases * len(decorator.args[1].elts)
            total += cases
    return total


# ---------------------------------------------------------------------------
# Reading the numbers out of the README
# ---------------------------------------------------------------------------

DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER = re.compile(
    r"(?<![\w.,/-])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e-?\d+)?(?!\w)|(?<=/)\d+(?!\w)"
)


def readme_prose():
    # Drop what is not prose: fenced code, inline code, link targets, HTML tags
    # and the "1." markers of the numbered build steps
    text = re.sub(r"```.*?```", "", README, flags=re.S)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s", "", text)
    return text


def readme_numbers():
    # Every number in reading order, with its line for the error message.
    # A date counts as one token.
    found = []
    for line_no, line in enumerate(readme_prose().splitlines(), start=1):
        position = 0
        while position < len(line):
            date = DATE.match(line, position)
            if date and (position == 0 or not line[position - 1].isalnum()):
                found.append((date.group(0), line_no, line))
                position = date.end()
                continue
            number = NUMBER.match(line, position)
            if number:
                found.append((number.group(0), line_no, line))
                position = number.end()
                continue
            position += 1
    return found


# ---------------------------------------------------------------------------
# The expected sequence, one entry per occurrence
# ---------------------------------------------------------------------------

FIGURE_NUMBERS = []     # (README text, figure file the notebook draws)
HISTORY_NUMBERS = []    # (README text, text in MODEL_CARD.md)


def result(text):
    # A pipeline result, already formatted from a committed output file
    return text


def setting(text):
    # A setting the code in src/ runs with, already formatted
    return text


def from_figure(text, figure, why):
    # A value read off a chart the notebook draws; the chart must be committed
    assert why
    FIGURE_NUMBERS.append((text, figure))
    return text


def from_history(text, needle):
    # A figure from an earlier version of the model, recorded in MODEL_CARD.md
    HISTORY_NUMBERS.append((text, needle))
    return text


def fixed(text, why):
    # A number that is part of the wording, not a result
    assert why
    return text


def pct(share, places=1):
    # 0.99564 -> "99.6"
    return f"{share * 100:.{places}f}"


def count(number):
    # 2754 -> "2,754"
    return f"{int(number):,}"


def bn(amount, places=2):
    # 4190812449.66 -> "4.19"
    return f"{amount / 1e9:.{places}f}"


def millions(amount, places):
    # 46474810.66 -> "46.5"
    return f"{amount / 1e6:.{places}f}"


def expected_numbers():
    # Sources, each formatted the way the README prints it
    data = NOTEBOOK_RESULTS["data"]
    split = NOTEBOOK_RESULTS["split"]
    stored = BUNDLE["metrics"]
    folds = SUMMARY["folds"]
    model_row = IMPACT[f"Model, threshold {HOLDOUT['operating_threshold']:.4f}"]
    lower_row = IMPACT[f"Model, threshold {IMPACT_FILE['comparison_threshold']}"]
    nothing = IMPACT["No screening"]
    builtin = IMPACT["isFlaggedFraud flag"]
    full_drain = SCENARIOS["drain_sweep"][-1]
    near_drain = SCENARIOS["drain_sweep"][-2]
    assert full_drain["drain_share"] == 1.0 and near_drain["drain_share"] == 0.99
    psi = psi_in_last_window()

    caught = HOLDOUT["frauds_caught"]
    frauds = HOLDOUT["holdout_frauds"]
    false_alarms = HOLDOUT["false_alarms"]
    recall = pct(caught / frauds)
    precision = pct(caught / (caught + false_alarms))
    rows = count(data["transactions"])
    holdout_rows = count(HOLDOUT["holdout_rows"])
    split_step = setting(str(config.SPLIT_STEP))
    fit_end = setting(str(BUNDLE["fit_steps"][1]))
    calibration_start = setting(str(BUNDLE["calibration_steps"][0]))
    last_hour = result(str(folds[-1]["test_end_step"]))
    pr_auc = f"{HOLDOUT['pr_auc']:.4f}"
    threshold = f"{HOLDOUT['operating_threshold']:.4f}"
    walk_pr_auc = f"{SUMMARY['PR-AUC']['mean']:.4f}"
    value_share = pct(model_row["share_of_fraud_value_stopped"], 2)
    builtin_share = pct(builtin["share_of_fraud_value_stopped"])
    last_precision = pct(folds[-1]["Precision"])
    miss_cost = setting(f"{features.COST_PER_MISSED_FRAUD:,}")
    alarm_cost = setting(f"{features.COST_PER_FALSE_ALARM:,}")
    cost_ratio = setting(f"{features.COST_PER_MISSED_FRAUD // features.COST_PER_FALSE_ALARM}")
    train_rate = pct(split["train_fraud_rate"], 2)
    test_rate = pct(split["test_fraud_rate"], 2)
    tuning_end = setting(str(config.TUNING_SPLITS[-1][1]))
    trees = result(str(BUNDLE["n_estimators"]))

    seq = []
    # Title line, badges and the headline chart's description
    seq += [result(f"{data['transactions'] / 1e6:.2f}")]
    seq += [fixed("3.12", "Python version badge")]
    seq += [result(builtin_share), result(value_share)]
    # What this project does
    seq += [rows, last_hour, result(f"{folds[-1]['test_end_step'] / 24:.0f}")]
    # Results: the windows, then the bullets
    seq += [fixed("1", "first hour"), fit_end, calibration_start, split_step, fixed("6", "step number")]
    seq += [holdout_rows, split_step]
    seq += [count(frauds)]
    seq += [result(pr_auc)]
    seq += [fixed("1.0", "PR-AUC of a perfect model")]
    seq += [count(caught), count(frauds), result(recall)]
    seq += [result(str(false_alarms)), result(precision)]
    seq += [result(pct(AT_REAL_RATE["precision"]))]
    seq += [result(value_share), result(builtin_share)]
    seq += [result(walk_pr_auc)]
    seq += [fixed("8", "step number"), result(pr_auc)]
    seq += [result(pct(SUMMARY["Precision"]["mean"])), result(last_precision)]
    seq += [fixed("10", "step number")]
    # Results table
    seq += [result(pr_auc), fixed("1.0", "PR-AUC of a perfect model")]
    seq += [result(recall), count(caught), count(frauds)]
    seq += [result(precision), result(str(false_alarms))]
    seq += [result(f"{stored['Brier']:.6f}"), fixed("0", "Brier score of a perfect model")]
    seq += [result(threshold)]
    seq += [result(str(SUMMARY["n_folds"])), result(walk_pr_auc), result(f"{SUMMARY['PR-AUC']['std']:.4f}")]
    seq += [result(pct(SUMMARY["Precision"]["mean"])), result(pct(SUMMARY["Recall"]["mean"])),
            result(str(SUMMARY["n_folds"]))]
    # Money table
    seq += [result(f"{nothing['fraud_value_stopped']:.0f}"), result(bn(nothing["fraud_value_missed"])),
            result(str(nothing["honest_transactions_frozen"]))]
    seq += [result(millions(builtin["fraud_value_stopped"], 1)), result(builtin_share),
            result(bn(builtin["fraud_value_missed"])), result(str(builtin["honest_transactions_frozen"]))]
    seq += [result(threshold), result(bn(model_row["fraud_value_stopped"], 3)), result(value_share),
            result(millions(model_row["fraud_value_missed"], 2)), result(str(model_row["honest_transactions_frozen"]))]
    seq += [result(str(IMPACT_FILE["comparison_threshold"])), result(bn(lower_row["fraud_value_stopped"], 3)),
            result(pct(lower_row["share_of_fraud_value_stopped"], 2)),
            result(millions(lower_row["fraud_value_missed"], 2)), result(str(lower_row["honest_transactions_frozen"]))]
    seq += [result(str(model_row["honest_transactions_frozen"])), result(millions(model_row["honest_value_frozen"], 2)),
            result(str(builtin["frauds_caught"])), count(frauds)]
    # Step 1: look at the data
    seq += [fixed("1", "notebook part number"), rows, result(count(data["frauds"]))]
    seq += [result(pct(data["fraud_rate"], 2))]
    pattern = data["balance_patterns_pct"]
    whole_balance = "amount equals the sender's whole balance"
    seq += [result(count(data["transfer_and_cash_out"])), result(f"{pattern['fraud'][whole_balance]:.2f}")]
    seq += [result(f"{pattern['honest'][whole_balance]:.1f}")]
    seq += [result(pct(1 - data["fraud_rate"], 2))]
    # Step 2: the built-in flag
    seq += [result(str(builtin["frauds_caught"])), count(frauds), result(builtin_share)]
    # Step 3: the balance fingerprint caption
    seq += [from_figure("0.99", "figs/balance_fingerprint.png", "honest: amount against the sender error"),
            from_figure("0.09", "figs/balance_fingerprint.png", "fraud: amount against the sender error"),
            from_figure("0.65", "figs/balance_fingerprint.png", "fraud: amount against the recipient error")]
    # Step 4: the time split
    seq += [fixed("1", "first hour"), split_step, fixed("1", "first hour"), fit_end, calibration_start, split_step,
            result(str(int(split_step) + 1)), last_hour]
    seq += [result(test_rate), result(train_rate)]
    # Step 5: tuning
    seq += [fixed("40", "Optuna trials, the default of src/tune.py --trials")]
    seq += [tuning_end]
    seq += [trees]
    # Step 6: calibration
    seq += [calibration_start, split_step]
    # Step 7: the threshold from two assumed costs
    seq += [miss_cost]
    seq += [alarm_cost]
    seq += [cost_ratio, result(threshold)]
    seq += [result(str(IMPACT_FILE["comparison_threshold"])), result(threshold), count(caught),
            result(str(false_alarms))]
    seq += [result(f"{near_drain['drain_share'] * 100:.0f}"), result(f"{near_drain['fraud_probability'] * 100:.0f}")]
    seq += [result(f"{full_drain['drain_share'] * 100:.0f}"), result(pct(full_drain["fraud_probability"]))]
    seq += [result(f"{SUMMARY['PR-AUC']['mean']:.3f}")]
    # Step 8: walk-forward windows
    window_hours = []
    for train_end, test_end in config.FOLDS:
        window_hours += [setting(str(train_end + 1)), setting(str(test_end))]
    seq += window_hours
    seq += [result(walk_pr_auc), result(str(folds[0]["threshold"])), result(str(folds[1]["threshold"])),
            result(str(folds[2]["threshold"]))]
    seq += [result(str(folds[3]["threshold"])), result(last_precision)]
    other_precisions = []
    for fold in folds[:-1]:
        other_precisions.append(fold["Precision"])
    seq += [result(pct(min(other_precisions))), result(f"{max(other_precisions) * 100:.0f}")]
    # Step 9: drift
    seq += [fixed("0.10", "conventional PSI line for stable, as in src/monitoring.py"),
            fixed("0.25", "conventional PSI line for investigate, as in src/monitoring.py")]
    seq += [setting("701"), last_hour, result(f"{psi['orig_drain_ratio']:.2f}"),
            result(f"{psi['dest_amount_ratio']:.2f}")]
    # Step 10: SHAP shares
    for entry in SHARES[:4]:
        seq += [result(f"{entry['importance_pct']:.2f}")]
    seq += [result(f"{SHARES[4]['importance_pct']:.2f}"),
            result(f"{SHARES[0]['importance_pct'] + SHARES[1]['importance_pct']:.0f}")]
    # Step 11: the export and the parity test
    seq += [fixed("20,000", "N_ROWS in tests/test_export_parity.py; checked below")]
    seq += [fixed("6.6e-08", "worst gap recorded in api/score.py; checked below")]
    # Problems I hit
    small = "12"
    assert SCENARIOS["small_full_drain"]["description"].startswith(small + "-unit")
    seq += [result(small), result(small)]
    seq += [fixed("100", "the first model's score in the 12-unit test, history")]
    seq += [result(f"{full_drain['drain_share'] * 100:.0f}")]
    seq += [result(pct(full_drain["fraud_probability"])), result(f"{near_drain['drain_share'] * 100:.0f}"),
            result(f"{near_drain['fraud_probability'] * 100:.4f}"), result(f"{full_drain['drain_share'] * 100:.0f}")]
    seq += [result(small)]
    seq += [result(f"{SCENARIOS['small_full_drain']['fraud_probability']:.4f}")]
    seq += [tuning_end]
    seq += [fixed("1.0", "recall of the leaked walk-forward windows, history")]
    seq += [from_history("99.85", "from 99.85% to 98.42%"), from_history("98.42", "from 99.85% to 98.42%")]
    seq += [result(BUNDLE["trained_on"]), trees, setting(f"{config.HOLDOUT_STEP_SHARE * 100:.0f}"),
            result(pct(stored["Precision"], 2))]
    seq += [result(f"{SCENARIOS['partial_skim']['fraud_probability'] * 100:.4f}")]
    # Data
    seq += [fixed("2016", "year of the PaySim paper"), rows, last_hour]
    seq += [result(pct(data["fraud_rate"], 2)), rows, result(train_rate), result(test_rate), holdout_rows]
    # Limits
    seq += [result(f"{NOTEBOOK_RESULTS['models']['Random forest']['PR-AUC']:.4f}")]
    seq += [result(f"{SCENARIOS['partial_skim']['fraud_probability'] * 100:.0f}")]
    seq += [result(test_rate), result(pct(AT_REAL_RATE["fraud_rate"], 2))]
    seq += [result(f"{AT_REAL_RATE['false_alarms']:.0f}"), result(pct(AT_REAL_RATE["precision"]))]
    seq += [result(str(SUMMARY["threshold"]["min"])), result(str(SUMMARY["threshold"]["max"]))]
    seq += [miss_cost, alarm_cost]
    # How to run and License
    seq += [fixed("3.12", "minimum Python version")]
    seq += [fixed("2016", "year of the PaySim paper")]
    return seq


def test_every_readme_number_in_order():
    expected = expected_numbers()
    found = readme_numbers()
    for index in range(min(len(expected), len(found))):
        text, line_no, line = found[index]
        assert text == expected[index], (
            f"number {index + 1} in the README is {text!r}, expected {expected[index]!r}; "
            f"line {line_no}: {line.strip()[:160]}"
        )
    if len(found) > len(expected):
        extra_text, extra_line, _ = found[len(expected)]
        raise AssertionError(f"README has a number the test does not expect: {extra_text!r} on line {extra_line}")
    assert len(found) == len(expected), f"README ends before the expected number {expected[len(found)]!r}"


def test_figure_numbers_come_from_committed_charts():
    expected_numbers()
    assert FIGURE_NUMBERS
    for text, figure in FIGURE_NUMBERS:
        assert (ROOT / figure).is_file(), f"README figure {text} cites {figure}, which is not committed"


def test_history_numbers_are_recorded_in_the_model_card():
    expected_numbers()
    assert HISTORY_NUMBERS
    for text, needle in HISTORY_NUMBERS:
        assert needle in MODEL_CARD, f"README figure {text} is not in MODEL_CARD.md ({needle!r})"


def test_fixed_numbers_match_their_files():
    # The fixed numbers above that name a file are checked against it here
    parity = (ROOT / "tests" / "test_export_parity.py").read_text(encoding="utf-8")
    assert "N_ROWS = 20_000" in parity
    scorer = (ROOT / "api" / "score.py").read_text(encoding="utf-8")
    assert "6.6e-08" in scorer
    tuner = (ROOT / "src" / "tune.py").read_text(encoding="utf-8")
    assert 'p.add_argument("--trials", type=int, default=40' in tuner


def test_metrics_stored_with_the_model_match_the_recomputed_file():
    stored = BUNDLE["metrics"]
    assert stored["PR-AUC"] == HOLDOUT["pr_auc"]
    assert stored["Frauds caught"] == HOLDOUT["frauds_caught"]
    assert stored["False alarms"] == HOLDOUT["false_alarms"]
    assert stored["Holdout rows"] == HOLDOUT["holdout_rows"]
    assert round(BUNDLE["operating_threshold"], 4) == HOLDOUT["operating_threshold"]
    assert BUNDLE["fit_steps"] == [1, 392] and BUNDLE["calibration_steps"] == [393, config.SPLIT_STEP]


def test_split_matches_the_notebook_and_the_holdout_file():
    split = NOTEBOOK_RESULTS["split"]
    assert split["split_step"] == config.SPLIT_STEP
    assert split["test_rows"] == HOLDOUT["holdout_rows"]
    assert split["test_frauds"] == HOLDOUT["holdout_frauds"]


def test_the_lower_threshold_gives_the_same_result():
    higher = IMPACT[f"Model, threshold {HOLDOUT['operating_threshold']:.4f}"]
    lower = IMPACT[f"Model, threshold {IMPACT_FILE['comparison_threshold']}"]
    assert lower["frauds_caught"] == higher["frauds_caught"] == HOLDOUT["frauds_caught"]
    assert lower["honest_transactions_frozen"] == higher["honest_transactions_frozen"] == HOLDOUT["false_alarms"]


def test_the_cost_curve_is_flat_from_half_to_the_threshold():
    # The README says every cut-off from 0.5 to 0.8852 gives the same holdout result
    expected_cost = (features.COST_PER_MISSED_FRAUD * HOLDOUT["frauds_missed"]
                     + features.COST_PER_FALSE_ALARM * HOLDOUT["false_alarms"])
    checked = 0
    with open(ROOT / "dashboard" / "data" / "threshold_cost_curve.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cut_off = float(row["threshold"])
            if 0.5 <= cut_off <= HOLDOUT["operating_threshold"]:
                assert int(row["cost"]) == expected_cost, f"cost at {cut_off} is {row['cost']}"
                checked += 1
    assert checked >= 30


def test_headline_sentences():
    # The headline sentences, pinned whole so a stray copy elsewhere cannot stand in
    caught = HOLDOUT["frauds_caught"]
    frauds = HOLDOUT["holdout_frauds"]
    recall = caught / frauds
    precision = caught / (caught + HOLDOUT["false_alarms"])
    model_row = IMPACT[f"Model, threshold {HOLDOUT['operating_threshold']:.4f}"]
    assert f"{recall:.1%}" == "99.6%" and f"{precision:.1%}" == "99.9%"
    assert f"{model_row['share_of_fraud_value_stopped']:.2%}" == "99.98%"
    assert f"**Ranking quality: PR-AUC {HOLDOUT['pr_auc']:.4f}.**" in README
    assert f"**It caught {caught:,} of the {frauds:,} frauds (recall {recall:.1%}).**" in README
    assert f"**It froze {HOLDOUT['false_alarms']} honest transfers (precision {precision:.1%}).**" in README
    assert f"**It stopped {model_row['share_of_fraud_value_stopped']:.2%} of the fraud value**" in README
    assert "**Every figure here comes from a simulator" in README


def test_readme_test_count_matches_tests_folder():
    stated = re.findall(r"(\d+) tests", README)
    assert len(stated) == 2, "README should state the test count under How to run and Project structure"
    actual = count_collected_tests()
    for number in stated:
        assert int(number) == actual, f"README says {number} tests, tests/ holds {actual}"
