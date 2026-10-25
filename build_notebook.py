"""Generate Fraud_Detection_System.ipynb.

This script is the source of the notebook: edit it, regenerate, and execute the
notebook. Never edit the .ipynb by hand.

    python build_notebook.py
    jupyter nbconvert --to notebook --execute Fraud_Detection_System.ipynb \
        --output Fraud_Detection_System.ipynb --ExecutePreprocessor.timeout=3600

The notebook is a guided build that a junior data scientist can follow cell by
cell and rebuild alone. It has three parts: 1. Prepare Data (import, explore,
split), 2. Build Model (baseline, iterate, evaluate) and 3. Communicate Results.
Every step is a numbered markdown cell (what to do, why, and one Concept line),
then one small code cell with a comment above each logical line and a printout
of what changed, then a "Check your work" cell whose asserts stop the run at the
step that went wrong. The expected values in the asserts come from an executed
run on the full PaySim file.

The numbers the notebook writes to outputs/notebook_results.json must not
change when the teaching text changes: compare the file leaf by leaf with the
previous run after any edit to this builder.
"""
import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =====================================================================
# TITLE
# =====================================================================
md("""
# Detecting Fraud in Mobile-Money Transfers: a Guided Build

**The question.** A mobile-money operator freezes a transfer when it looks fraudulent. Which transfers are
fraudulent, how much of the fraud value can a model stop, and how many honest customers does it freeze to do
so?

**The data.** PaySim, a public simulator built from one month of logs of a mobile-money service in an African
country (Lopez-Rojas, Elmir and Axelsson, 2016, on [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)):
6,362,620 simulated transactions, one row each, over 743 simulated hours. *Mobile money* means a balance held on
a phone account and moved by text message or app; a *fraud* in PaySim is a transfer out of a victim's account by
someone who has taken it over.

**What you will build.** You will load the file with a plain `pd.read_csv`, check it, explore it, turn one
accounting rule into features, split it by time, and then build up from a model that never flags anything, to a
two-question tree, to XGBoost and three other model families. You will choose a cut-off from the cost of
mistakes, test the model period by period, explain single decisions, and finish with the fraud value each screen
stops and the honest customers it freezes.

**The finished result.** On the 132,136 test transactions after hour 490, XGBoost with a class weight, the
model this notebook carries forward, reaches a PR-AUC of 0.9989. At the cheapest cut-off it stops 99.90% of the fraud
value and freezes 2 honest customers; PaySim's own built-in flag stops 1.11%. The shipped model in `src/`, built
on the same ideas with fewer inputs, catches 2,742 of 2,754 frauds and freezes 3 honest transfers (Step 3.6 reads
those figures from its output files).

**How to use this notebook.** Run it from the top. Each step has three parts: a short explanation with one
**Concept** line, one small code cell, and a **Check your work** cell. If a Check cell fails, the step above it
went wrong; fix that step before going on. Step 3.7 records how long the run took, and it needs several GB of
free memory.

**PaySim is a simulator, so how well any model here would work on a real operator's transfers is unproven.**
Every score below says more about the method than about the accuracy to expect on a real network (see
Limitations at the end).
""")

# =====================================================================
# PART 1
# =====================================================================
md("""
---
# Part 1: Prepare Data

Part 1 has three stages. **Import** reads the file and checks it is complete. **Explore** asks what the data
can support: where fraud happens, how rare it is, and what it looks like in the balances. **Split** turns the
accounting rule frauds break into features, guards against leakage, and splits the rows by time so every model
is tested on the future.

## 1A. Import

### Step 1.1: Load the libraries

Every tool the notebook uses comes from one of these libraries. Models are imported later, in the step that
first uses them, so each new idea arrives on its own.

**Concept: why import at the top?** A reader can see in one place what the notebook depends on, and a missing
library fails here, in the first cell, instead of half an hour into the run.
""")

code("""
# os: create the folders for charts and results
import os
# sys: let Python find the project's own code in src/ (Step 3.6)
import sys
# time: the clock that times the whole run
import time
# warnings: hide library notices about future versions
import warnings

# matplotlib: draw and save every chart
import matplotlib.pyplot as plt
# seaborn: chart styling and the heatmaps
import seaborn as sns

# numpy: fast arrays of numbers
import numpy as np
# pandas: the tables the whole notebook works on
import pandas as pd

# The measures used to score every model
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

# Print the versions, so a reader can match them if a number differs
print("pandas", pd.__version__, "| numpy", np.__version__)
""")

code("""
# Check your work
# pandas 2 or later, the version family this notebook was written for
assert int(pd.__version__.split(".")[0]) >= 2, f"expected pandas 2 or later, got {pd.__version__}"
# The scoring functions imported
assert callable(average_precision_score) and callable(precision_recall_curve), "the scoring functions did not import"
""")

md("""
### Step 1.2: Fix the settings every later step refers to

Put the numbers you might want to change in one cell, with a comment saying what each one does.

**Concept: what is a random seed?** Several steps draw random samples or build random trees. Fixing the seed
makes every run draw the same ones, so the same code always prints the same numbers.
""")

code("""
# Start a clock, so the results file can record how long the run took
RUN_STARTED = time.time()

# One seed, so every run gives the same numbers
SEED = 42

# Train on hours 1 to 490, test on hours 491 to 743 (Step 1.27)
SPLIT_STEP = 490

# A missed fraud costs as much as 10 false alarms (Step 2.14)
COST_RATIO = 10

# Threads for the boosted models; more threads use more memory, the results are the same
N_JOBS = 1

# Random forests share one copy of the data, so extra threads are cheap
RF_JOBS = 4

# Seed numpy's random generator
np.random.seed(SEED)

# Print the settings, so the output records the values this run used
print("SEED", SEED, "| SPLIT_STEP", SPLIT_STEP, "| COST_RATIO", COST_RATIO, "| N_JOBS", N_JOBS, "| RF_JOBS", RF_JOBS)
""")

code("""
# Check your work
# The seed every later step relies on
assert SEED == 42, f"expected seed 42, got {SEED}"
# The teaching cost ratio used in Step 2.14
assert COST_RATIO == 10, f"expected a cost ratio of 10, got {COST_RATIO}"
""")

md("""
### Step 1.3: Set how tables and charts look, and make the output folders

**Concept: why save figures to a folder?** The README shows some of these charts. Saving them from code means
the picture in the README is always the one this notebook drew.
""")

code("""
# Show decimals with thousands separators and two places
pd.options.display.float_format = "{:,.2f}".format

# A light chart style with grid lines
sns.set_theme(style="whitegrid", context="notebook")
# Every chart 10 by 5 inches unless a step says otherwise
plt.rcParams["figure.figsize"] = (10, 5)

# Hide notices about future library versions
warnings.filterwarnings("ignore", category=FutureWarning)
# Hide notices libraries raise about their settings; they do not change the results
warnings.filterwarnings("ignore", category=UserWarning)

# A folder for the charts
os.makedirs("figs", exist_ok=True)
# A folder for the results file
os.makedirs("outputs", exist_ok=True)
# Confirm both folders exist
print("figs/ exists:", os.path.isdir("figs"), "| outputs/ exists:", os.path.isdir("outputs"))
""")

code("""
# Check your work
# The split hour is 490
assert SPLIT_STEP == 490, f"the split should be at hour 490, got {SPLIT_STEP}"
# Both output folders exist
assert os.path.isdir("figs") and os.path.isdir("outputs"), "the output folders were not created"
""")

md("""
### Step 1.4: Read the PaySim file

The CSV (470 MB) is not stored in the repository. Download "PaySim 1" from
[Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and put `PS_20174392719_1491204439457_log.csv` in the
`data` folder next to this notebook.

**Concept: what is a DataFrame?** A pandas table: rows are transactions, columns are fields, and every column
has one type (whole numbers, decimals or text).
""")

code("""
# Read the whole file into one table
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")

# The object type: a pandas DataFrame
print(type(df))
# Rows and columns
print("rows, columns:", df.shape)
# Memory the table takes
print(f"memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
# The first five rows
df.head()
""")

code("""
# Check your work
# The full file: 6,362,620 rows and 11 columns
assert df.shape == (6_362_620, 11), f"expected 6,362,620 rows and 11 columns, got {df.shape}"
""")

md("""
### Step 1.5: Keep two raw rows aside for later

Rows 2 and 3 are the first two frauds in the file. Step 1.9 works through them by hand, and Step 3.6 runs the
shipped feature code in `src/features.py` on them, which expects PaySim's original column names. So copy them now,
before the names change.

**Concept: why `.copy()`?** Without it pandas may hand back a view of the big table, and a later change to
either one could change the other.
""")

code("""
# Copy rows 2 and 3 with their original column names
raw_examples = df.loc[[2, 3]].copy()
# Type and size of the copy
print(type(raw_examples), raw_examples.shape)
# Show the two kept rows
raw_examples
""")

code("""
# Check your work
# Both kept rows are frauds
assert raw_examples["isFraud"].tolist() == [1, 1], "rows 2 and 3 should both be fraud"
""")

md("""
### Step 1.6: Give the balance columns one naming pattern

PaySim's names are inconsistently capitalised (`oldbalanceOrg` but `newbalanceOrig`). Rename the four balance
columns so the sender's are `...Orig` and the recipient's are `...Dest`.

**Concept: sender and recipient.** *Orig* is the account the money leaves (the originator); *Dest* is the
account it goes to (the destination).
""")

code("""
# Old name -> new name, one pair per balance column
new_names = {
    "oldbalanceOrg": "oldBalanceOrig",
    "newbalanceOrig": "newBalanceOrig",
    "oldbalanceDest": "oldBalanceDest",
    "newbalanceDest": "newBalanceDest",
}

# Apply the new names to the table
df = df.rename(columns=new_names)
# Print the column names to confirm the change
print(list(df.columns))
""")

code("""
# Check your work
# The new name is there and the old one is gone
assert "oldBalanceOrig" in df.columns and "oldbalanceOrg" not in df.columns, "the rename did not apply"
""")

md("""
Each row is one transaction:

| Column | Meaning |
|---|---|
| `step` | Hour of the simulation, 1 to 743 (about 31 days) |
| `type` | TRANSFER, CASH_OUT, PAYMENT, CASH_IN or DEBIT |
| `amount` | Value of the transaction, in the simulator's currency units |
| `nameOrig`, `nameDest` | Sender and recipient account IDs (`C` customer, `M` merchant) |
| `oldBalanceOrig`, `newBalanceOrig` | Sender's balance before and after |
| `oldBalanceDest`, `newBalanceDest` | Recipient's balance before and after |
| `isFraud` | 1 if the transaction was fraud: the label to predict |
| `isFlaggedFraud` | 1 if PaySim's own built-in rule flagged it |

### Step 1.7: Look for missing values and duplicates

**Concept: data-quality check.** Before any analysis, confirm the table is complete: a missing value or a
duplicated row would quietly change every count after it.
""")

code("""
# Count missing cells across the whole table
missing_cells = int(df.isna().sum().sum())

# Count rows that repeat an earlier row exactly
duplicated_rows = int(df.duplicated().sum())

# Print the number of missing cells
print("missing cells:", missing_cells)
# Print the number of duplicated rows
print("duplicated rows:", duplicated_rows)
""")

code("""
# Check your work
# No missing cells
assert missing_cells == 0, f"expected no missing cells, got {missing_cells}"
# No duplicated rows
assert duplicated_rows == 0, f"expected no duplicated rows, got {duplicated_rows}"
""")

md("""
### Step 1.8: Look for impossible amounts, negative balances and missing hours

**Concept: a sanity rule.** Money amounts are above zero, balances are never negative, and a month of hourly
steps should have every hour present. A row that breaks one of these is either an error or something to explain.
""")

code("""
# The four balance columns, used in several later steps
balance_cols = ["oldBalanceOrig", "newBalanceOrig", "oldBalanceDest", "newBalanceDest"]

# True where the amount is zero or less
zero_amount = df["amount"] <= 0
# How many of those rows are fraud
zero_amount_frauds = int((zero_amount & (df["isFraud"] == 1)).sum())

# Rows where any balance is negative
negative_balance_rows = int((df[balance_cols] < 0).any(axis=1).sum())

# Print the zero amounts and their frauds
print(f"amount <= 0: {int(zero_amount.sum())} (of which fraud: {zero_amount_frauds})")
# Print the rows with a negative balance
print("rows with a negative balance:", negative_balance_rows)
# Print the first and last hour and how many distinct hours there are
print(f"hours: {df['step'].min()} to {df['step'].max()}, {df['step'].nunique()} distinct")
""")

code("""
# Check your work
# No balance is negative
assert negative_balance_rows == 0, f"expected no negative balances, got {negative_balance_rows}"
# 16 zero amounts, all fraud
assert int(zero_amount.sum()) == 16 and zero_amount_frauds == 16, "expected 16 zero amounts, all fraud"
# Every hour from 1 to 743 is present
assert df["step"].nunique() == 743 and df["step"].min() == 1, "expected every hour from 1 to 743"
""")

md("""
The file is clean in the usual sense: nothing missing, no duplicates, no negative balances, all 743 hours
present. The 16 transactions with an amount of zero are all fraud: attempted transfers of nothing. They are kept,
because they are labelled, and Part 3 counts fraud by value, where they weigh nothing.

### Step 1.9: Read two frauds as an accountant would

Before any statistics, look closely at rows 2 and 3.

**Concept: the ledger rule.** In an honest transfer the sender's balance falls by exactly the amount and the
recipient's rises by exactly the amount. Accountants call a gap between the two sides of an entry a
*discrepancy*.
""")

code("""
# The columns that tell the story of one transaction
story_cols = ["type", "amount", "oldBalanceOrig", "newBalanceOrig", "oldBalanceDest", "newBalanceDest", "isFraud"]

# Show rows 2 and 3
df.loc[[2, 3], story_cols]
""")

code("""
# Check your work
# Both rows move 181 and are fraud
assert df.loc[[2, 3], "amount"].tolist() == [181.0, 181.0], "rows 2 and 3 should each move 181"
# Row 3's recipient had 21,182 before and 0 after
assert df.loc[3, "oldBalanceDest"] == 21_182 and df.loc[3, "newBalanceDest"] == 0, "row 3's recipient balances changed"
""")

md("""
- **Row 2, a TRANSFER of 181.** The sender had 181 and ends with 0, so the whole balance left. The recipient
  had 0 and still has 0: 181 left one account and arrived nowhere.
- **Row 3, a CASH_OUT of 181.** The sender again goes from 181 to 0. The recipient had 21,182 and ends with 0,
  so the recipient's balance *fell* while receiving money.

Both frauds break the ledger rule on the recipient's side. Keep these two rows in mind; Step 1.23 turns the
observation into features.

## 1B. Explore

### Step 1.10: Count transactions, value and frauds by type

**Concept: group and summarise.** `groupby` splits the table by one column and `agg` computes one summary per
group, here a count, a sum, a median and a fraud count.
""")

code("""
# One row per transaction type: how many, how much money, the typical amount, and how many frauds
by_type = df.groupby("type").agg(transactions=("amount", "size"), value=("amount", "sum"),
                                 median_amount=("amount", "median"), frauds=("isFraud", "sum"))

# The share of each type's transactions that are fraud
by_type["fraud_rate"] = by_type["frauds"] / by_type["transactions"]

# Type and size of the summary
print(type(by_type), by_type.shape)
# Show it, most frauds first
by_type.sort_values("frauds", ascending=False)
""")

code("""
# Check your work
# The three other types have no fraud
assert by_type.loc[["CASH_IN", "DEBIT", "PAYMENT"], "frauds"].sum() == 0, "fraud should occur only in TRANSFER and CASH_OUT"
# All 8,213 frauds are counted
assert int(by_type["frauds"].sum()) == 8_213, f"expected 8,213 frauds, got {int(by_type['frauds'].sum())}"
""")

md("""
### Step 1.11: Chart the count and value by type

Draw the two columns of the table side by side, so the common types and the money-moving types can be compared.

**Concept: count against value.** A type can be common but move little money, or rare but move a lot. Fraud
losses are counted in money, so both views matter.
""")

code("""
# Two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Left: number of transactions, in millions
(by_type["transactions"] / 1e6).sort_values().plot(kind="barh", ax=axes[0], color="#4a7ab5")
# Title of the transactions panel
axes[0].set_title("Transactions by type (millions)")
# Label the transactions axis in millions
axes[0].set_xlabel("Transactions (millions)")

# Right: money moved, in billions
(by_type["value"] / 1e9).sort_values().plot(kind="barh", ax=axes[1], color="#4a7ab5")
# Title of the value panel
axes[1].set_title("Value moved by type (billions)")
# Label the value axis in billions
axes[1].set_xlabel("Value (billions of simulated units)")

# Keep the two type panels from overlapping
plt.tight_layout()
# Save the type chart as figs/transactions_by_type.png
plt.savefig("figs/transactions_by_type.png", dpi=150, bbox_inches="tight")
# Show the type chart
plt.show()
""")

code("""
# Check your work
# The type chart was saved for the README
assert os.path.isfile("figs/transactions_by_type.png"), "figs/transactions_by_type.png was not written"
""")

md("""
CASH_OUT and PAYMENT are the most common types, but TRANSFER moves the most money: its median transaction
(486,308) is about three times a CASH_OUT's (147,072). Fraud appears only in TRANSFER (4,097) and CASH_OUT
(4,116): in PaySim a fraud transfers money out of a victim's account and then cashes it out. The other three
types contain no fraud, so Step 1.20 sets them aside.

### Step 1.12: Count the frauds

**Concept: class imbalance.** When one outcome is far rarer than the other, the data is *imbalanced*. Most of
the choices in Part 2 exist because of it.
""")

code("""
# Number of frauds in the whole file
n_fraud = int(df["isFraud"].sum())

# Their total value, in billions
fraud_value_billions = df.loc[df["isFraud"] == 1, "amount"].sum() / 1e9

# The count is a plain Python int
print(type(n_fraud))
# Print the count, its share and its value
print(f"frauds: {n_fraud:,} of {len(df):,} transactions ({n_fraud / len(df):.4%}), worth {fraud_value_billions:.1f} billion")
""")

code("""
# Check your work
# 8,213 frauds in the file
assert n_fraud == 8_213, f"expected 8,213 frauds, got {n_fraud}"
""")

md("""
About 1 transaction in 775 is fraud.

### Step 1.13: See why accuracy misleads

Score the laziest possible detector: it says *not fraud* for every transaction.

**Concept: accuracy, recall and precision.** *Accuracy* is the share of all transactions labelled correctly.
*Recall* is frauds caught divided by all frauds: of the real fraud, how much was caught? *Precision* is frauds
caught divided by all transactions flagged: when an alarm is raised, how often is it real? If a model flags 100
transactions, 90 of them are fraud, and there were 120 frauds in total, precision is 90/100 = 90% and recall is
90/120 = 75%.
""")

code("""
# A prediction of 0 (not fraud) for every row
lazy_prediction = np.zeros(len(df), dtype=int)

# Share of rows where the prediction equals the label
accuracy = (lazy_prediction == df["isFraud"]).mean()

# Frauds the lazy detector caught
lazy_frauds_caught = int(((lazy_prediction == 1) & (df["isFraud"] == 1)).sum())

# Print the accuracy and the frauds caught
print(f"accuracy: {accuracy:.2%}   frauds caught: {lazy_frauds_caught}")
""")

code("""
# Check your work
# 99.87% accurate
assert round(accuracy * 100, 2) == 99.87, f"expected 99.87% accuracy, got {accuracy:.4%}"
# Yet it catches no fraud
assert lazy_frauds_caught == 0, "the lazy detector should catch nothing"
""")

md("""
99.87% accurate, and it catches nothing. Accuracy is the wrong yardstick for rare events, so this notebook
never uses it again. Recall exposes the lazy detector at once: it is 0.

### Step 1.14: Compare the amounts of fraud and honest transfers

Look only at TRANSFER and CASH_OUT, and use a log scale because the amounts span seven orders of magnitude.

**Concept: log scale.** `log10(amount)` turns 100 into 2 and 1,000,000 into 6, so small and huge amounts fit on
one axis.
""")

code("""
# True for TRANSFER and CASH_OUT rows, the only types with fraud
active_types = df["type"].isin(["TRANSFER", "CASH_OUT"])

# log10 of the amount (plus 1, so an amount of 0 does not break the log)
log_amount = np.log10(df.loc[active_types, "amount"] + 1)

# True where the row is fraud, for the TRANSFER and CASH_OUT rows only
is_fraud = df.loc[active_types, "isFraud"] == 1

# Type and length of the log amounts, and the frauds among them
print(type(log_amount), log_amount.shape, "| frauds among them:", int(is_fraud.sum()))
""")

code("""
# A blank chart for the amount histograms
fig, ax = plt.subplots(figsize=(10, 4.5))
# Histogram of the honest amounts
ax.hist(log_amount[~is_fraud], bins=80, density=True, alpha=0.6, label="honest")
# Histogram of the fraud amounts, on the same axis
ax.hist(log_amount[is_fraud], bins=80, density=True, alpha=0.6, label="fraud")

# Title of the amount chart
ax.set_title("Amount of TRANSFER and CASH_OUT transactions, fraud against honest")
# Label the x-axis with the log scale used
ax.set_xlabel("log10(amount + 1)")
# Label the y-axis: share of rows per bin
ax.set_ylabel("Density")
# Key for the honest and fraud colours
ax.legend()

# Keep the amount chart inside its margins
plt.tight_layout()
# Save it as figs/amount_distribution.png
plt.savefig("figs/amount_distribution.png", dpi=150, bbox_inches="tight")
# Show the amount chart
plt.show()
""")

code("""
# Summary statistics of the amount, honest (0) against fraud (1)
amount_summary = df.loc[active_types].groupby("isFraud")["amount"].describe()

# Frauds sitting exactly on PaySim's 10,000,000 cap for a single fraudulent transfer
frauds_at_cap = int(((df["amount"] == 10_000_000) & (df["isFraud"] == 1)).sum())

# Print how many frauds sit on the cap
print("frauds of exactly 10,000,000:", frauds_at_cap)
# Show the summary table
amount_summary
""")

code("""
# Check your work
# 287 frauds sit on the cap
assert frauds_at_cap == 287, f"expected 287 frauds at the cap, got {frauds_at_cap}"
# No fraud is above the cap
assert amount_summary.loc[1, "max"] == 10_000_000, "the largest fraud should be the 10,000,000 cap"
""")

md("""
Fraudulent transfers are larger in the middle (a median of about 441,000 against 171,000) but the two
distributions overlap heavily, so the amount alone cannot separate them. The fraud distribution stops dead at
10,000,000: the simulator caps a single fraudulent transfer there, and 287 frauds sit exactly on the cap.

### Step 1.15: Plot transactions and frauds per simulated day

**Concept: base rate.** The *base rate* is the share of fraud in a group of transactions. If honest volume
falls while fraud stays steady, the base rate rises, and every precision figure changes with it.
""")

code("""
# Day number for each transaction: hours 1-24 are day 1, 25-48 day 2, and so on
day = (df["step"] - 1) // 24 + 1

# Transactions and frauds per day
per_day = df.groupby(day).agg(transactions=("amount", "size"), frauds=("isFraud", "sum"))
# Type and size of the daily table
print(type(per_day), per_day.shape)
# The first five days
per_day.head()
""")

code("""
# A wide blank chart for the daily counts
fig, ax = plt.subplots(figsize=(12, 4.5))
# Bars: transactions per day, in thousands
ax.bar(per_day.index, per_day["transactions"] / 1e3, color="#4a7ab5", label="transactions (thousands)")
# Label the x-axis with the day
ax.set_xlabel("Simulated day")
# Label the left y-axis with the transactions
ax.set_ylabel("Transactions (thousands)")

# A second y-axis on the right for the frauds
ax2 = ax.twinx()
# Line: frauds per day
ax2.plot(per_day.index, per_day["frauds"], color="#c0504d", marker="o", label="frauds")
# Label the right y-axis with the frauds
ax2.set_ylabel("Frauds")
# Leave headroom above the highest fraud count
ax2.set_ylim(0, per_day["frauds"].max() * 1.3)
# Turn off the second axis's grid so the two grids do not clash
ax2.grid(False)

# Title saying what the chart shows
ax.set_title("Honest volume collapses after day 16; fraud continues at the same pace")
# Key for the bars, top left
ax.legend(loc="upper left")
# Key for the line, top right
ax2.legend(loc="upper right")
# Keep the daily chart inside its margins
plt.tight_layout()
# Save it as figs/volume_over_time.png
plt.savefig("figs/volume_over_time.png", dpi=150, bbox_inches="tight")
# Show the daily chart
plt.show()
""")

code("""
# Rows at or before the split hour
before_split = df[df["step"] <= SPLIT_STEP]
# Rows after it
after_split = df[df["step"] > SPLIT_STEP]
# Transactions per hour before the split, averaged over the hours
transactions_per_hour_before = before_split.groupby("step").size().mean()
# The same average after the split
transactions_per_hour_after = after_split.groupby("step").size().mean()

# Frauds per hour before the split
frauds_per_hour_before = before_split.groupby("step")["isFraud"].sum().mean()
# Frauds per hour after the split
frauds_per_hour_after = after_split.groupby("step")["isFraud"].sum().mean()

# Print the hourly transactions
print(f"transactions per hour, hours 1-{SPLIT_STEP}: {transactions_per_hour_before:,.0f}; after: {transactions_per_hour_after:,.0f}")
# Print the hourly frauds
print(f"frauds per hour, hours 1-{SPLIT_STEP}: {frauds_per_hour_before:.1f}; after: {frauds_per_hour_after:.1f}")
# Free the two slices, which only this step needs
del before_split, after_split
""")

code("""
# Check your work
# Hourly transaction volume before and after the split
assert round(transactions_per_hour_before) == 12_336 and round(transactions_per_hour_after) == 1_257, "transactions per hour changed"
# Hourly frauds before and after the split
assert round(frauds_per_hour_before, 1) == 11.1 and round(frauds_per_hour_after, 1) == 10.9, "frauds per hour changed"
""")

md("""
Honest volume averages 12,336 transactions an hour up to hour 490 and 1,257 after it, while fraud runs at about
11 an hour throughout (11.1 before, 10.9 after). That is why the test window in Step 1.27 has about ten times the
fraud rate of the training window: not more fraud, but far fewer honest transactions around it.

### Step 1.16: Plot the fraud rate by hour of day

**Concept: a time-of-day feature.** The hour of the day (0 to 23) repeats every day, so a model can learn from
it without learning the calendar.
""")

code("""
# Hour of the day for each active row
hour = df.loc[active_types, "step"] % 24

# Share of active rows that are fraud, per hour of day, in percent
by_hour = is_fraud.groupby(hour).mean() * 100
# Type and length: one value per hour of day
print(type(by_hour), by_hour.shape)
# The first eight hours, rounded
print(by_hour.round(1).head(8).to_string())
""")

code("""
# A blank chart for the hourly fraud rate
fig, ax = plt.subplots(figsize=(10, 4.5))
# A line with one point per hour
by_hour.plot(marker="o", color="#c0504d", ax=ax)
# Title of the hourly chart
ax.set_title("Share of TRANSFER and CASH_OUT transactions that are fraud, by hour of day")
# Label the x-axis with the hour
ax.set_xlabel("Hour of day")
# Label the y-axis with the fraud rate
ax.set_ylabel("Fraud rate (%)")
# Keep the hourly chart inside its margins
plt.tight_layout()
# Save it as figs/fraud_rate_by_hour.png
plt.savefig("figs/fraud_rate_by_hour.png", dpi=150, bbox_inches="tight")
# Show the hourly chart
plt.show()
""")

code("""
# Check your work
# More than half of 4 and 5 a.m. transfers are fraud
assert by_hour.loc[4] > 50 and by_hour.loc[5] > 50, "expected more than half of 4 and 5 a.m. transfers to be fraud"
""")

md("""
The rate jumps in the early hours: at 4 and 5 in the morning more than half of these transactions are fraud.
Fraud keeps going around the clock while honest activity drops off at night. The hour of day is therefore kept
as a model input in Step 1.21.

### Step 1.17: Check who sends and who receives

Account IDs start with `C` for a customer and `M` for a merchant.

**Concept: a mule account.** In real fraud, stolen money is often passed through *mule* accounts that receive
from many victims. If PaySim had them, repeat recipients would be a strong signal.
""")

code("""
# True where the recipient is a merchant
to_merchant = df["nameDest"].str.startswith("M")

# True where the row is fraud
fraud_rows = df["isFraud"] == 1

# Share paid to a merchant, by type, in percent
merchant_share_by_type = (to_merchant.groupby(df["type"]).mean() * 100).round(1)
# Print the merchant share for each type
print(merchant_share_by_type.to_string())
# Print how many frauds went to a merchant
print("frauds paid to a merchant:", int(to_merchant[fraud_rows].sum()))
""")

code("""
# Number of distinct sender accounts
distinct_senders = df["nameOrig"].nunique()
# Senders that appear in more than one transaction
repeat_senders = int((df["nameOrig"].value_counts() > 1).sum())

# Fraud recipients that also appear as a sender
fraud_recipients_who_send = int(df.loc[fraud_rows, "nameDest"].isin(df["nameOrig"]).sum())
# Fraud recipients that receive more than one fraud
repeat_fraud_recipients = int((df.loc[fraud_rows, "nameDest"].value_counts() > 1).sum())

# Print the sender counts
print(f"distinct senders: {distinct_senders:,} of {len(df):,} transactions; seen more than once: {repeat_senders:,}")
# Print the fraud recipients that also send
print(f"fraud recipients that appear as a sender anywhere: {fraud_recipients_who_send} of {n_fraud:,}")
# Print the fraud recipients hit more than once
print(f"fraud recipients that receive more than one fraud: {repeat_fraud_recipients}")
""")

code("""
# Check your work
# No fraud goes to a merchant
assert int(to_merchant[fraud_rows].sum()) == 0, "no fraud should go to a merchant"
# 9,298 senders appear more than once
assert repeat_senders == 9_298, f"expected 9,298 repeat senders, got {repeat_senders}"
# 18 fraud recipients also send; 44 receive more than one fraud
assert fraud_recipients_who_send == 18 and repeat_fraud_recipients == 44, "fraud recipient counts changed"
""")

md("""
Only PAYMENT goes to merchants, and no fraud does. Of 6,362,620 transactions, only 9,298 senders appear more
than once; 18 fraud recipients ever send anything and 44 receive a second fraud. So the account IDs carry almost
no pattern a model could learn from this data (a real network would be different), and Step 1.20 drops them.

### Step 1.18: Score the detector PaySim already provides

PaySim ships a rule-based flag, `isFlaggedFraud`. Any model built here must beat it.

**Concept: a benchmark.** A benchmark is an existing method you must beat for your work to be worth using. Here
the benchmark is the built-in flag.
""")

code("""
# True where PaySim's own rule flagged the transaction
flagged = df["isFlaggedFraud"] == 1

# Frauds among the flagged
caught = int((flagged & fraud_rows).sum())

# Print the flagged count and the frauds among them
print(f"transactions flagged: {int(flagged.sum())}   frauds among them: {caught}")
# Print the share of all frauds the flag catches
print(f"recall of the built-in flag: {caught / n_fraud:.2%}")
""")

code("""
# Check your work
# 16 flagged, all 16 fraud
assert int(flagged.sum()) == 16 and caught == 16, f"expected 16 flagged, 16 fraud; got {int(flagged.sum())}, {caught}"
""")

md("""
The built-in rule flags 16 transactions and catches 0.19% of the fraud. It is kept only as a benchmark for
Part 3, and dropped from the model inputs.

### Step 1.19: Read the balance patterns

How often does a balance read 0 before *and* after a non-zero transaction? That is almost certainly "not
recorded" rather than a real zero. Compare fraud and honest transfers on both sides, and check the pattern rows 2
and 3 share: the amount equals the sender's whole balance.

**Concept: a missing value in disguise.** A zero that should not be zero is a missing value written as a number.
Treating it as a real zero would teach a model something false.
""")

code("""
# Only the TRANSFER and CASH_OUT rows
active = df[active_types]

# Sender balance 0 before and after a non-zero amount
sender_zero_pair = (active["oldBalanceOrig"] == 0) & (active["newBalanceOrig"] == 0) & (active["amount"] > 0)

# Recipient balance 0 before and after a non-zero amount
recipient_zero_pair = (active["oldBalanceDest"] == 0) & (active["newBalanceDest"] == 0) & (active["amount"] > 0)

# The amount equals the sender's whole balance
whole_balance = active["oldBalanceOrig"] == active["amount"]

# The same fraud flag, for the balance patterns below
fraud_active = active["isFraud"] == 1
# Type and size of the active table
print(type(active), active.shape)
""")

code("""
# One row per pattern, one column each for fraud and honest: the share of rows showing it
balance_patterns = pd.DataFrame({
    "fraud": [sender_zero_pair[fraud_active].mean(), recipient_zero_pair[fraud_active].mean(), whole_balance[fraud_active].mean()],
    "honest": [sender_zero_pair[~fraud_active].mean(), recipient_zero_pair[~fraud_active].mean(), whole_balance[~fraud_active].mean()],
}, index=["sender balance 0 -> 0", "recipient balance 0 -> 0", "amount equals the sender's whole balance"])

# Show as percentages
(balance_patterns * 100).round(2)
""")

code("""
# Check your work
# The patterns as percentages, rounded as printed
pattern_pct = (balance_patterns * 100).round(2)
# 97.82% of frauds move the sender's whole balance
assert pattern_pct.loc["amount equals the sender's whole balance", "fraud"] == 97.82, "whole-balance share for fraud changed"
# 49.56% of frauds leave the recipient at 0 -> 0
assert pattern_pct.loc["recipient balance 0 -> 0", "fraud"] == 49.56, "recipient 0 -> 0 share for fraud changed"
""")

md("""
Three facts, all read off the balances:

- **The recipient's balance is unrecorded for 49.56% of frauds**, against 0.06% of honest transfers. Step 1.22
  marks that pattern rather than filling it in.
- **The sender's balance is unrecorded for 47.37% of honest transfers** (0.30% of frauds). This is a quirk of the
  simulator, so "the sender's balance went to zero" is not, on its own, a fraud signal.
- **97.82% of frauds take the sender's whole balance**, to the unit, against 0.00% of honest transfers. That is the
  signature PaySim's fraud generator leaves, and Step 2.27 shows the model finding it.

## 1C. Split

### Step 1.20: Keep TRANSFER and CASH_OUT, and set the label aside

Every fraud is in these two types (Step 1.10), so the other three can go. Separate the label, the thing to
predict, from the inputs.

**Concept: label and features.** The *label* (`isFraud`) is the answer. *Features* are the numbers the model
looks at to guess it. A model must never see the label among its features.
""")

code("""
# The label: what the model predicts
target = active["isFraud"]

# PaySim's built-in flag, kept aside for Part 3 only
builtin_flag = active["isFlaggedFraud"]

# The inputs: everything except the label, the built-in flag and the account IDs
features = active.drop(columns=["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"]).copy()
# Type and size of the input table
print(type(features), features.shape)
# Print how many rows and frauds were kept
print(f"kept {len(active):,} of {len(df):,} transactions, and all {int(target.sum()):,} frauds")
""")

code("""
# Check your work
# Every fraud survives the filter
assert int(target.sum()) == n_fraud, "filtering removed a fraud"
# 2,770,409 TRANSFER and CASH_OUT rows
assert len(features) == 2_770_409, f"expected 2,770,409 rows, got {len(features):,}"
""")

md("""
### Step 1.21: Turn the type into a number and add the hour of day

**Concept: encoding.** Most models take numbers only. With two types, one column of 0 (TRANSFER) and 1
(CASH_OUT) is enough.
""")

code("""
# TRANSFER -> 0, CASH_OUT -> 1, stored as a small integer
features["type"] = features["type"].map({"TRANSFER": 0, "CASH_OUT": 1}).astype("int8")

# Hour of the day, 0 to 23
features["hour"] = (features["step"] % 24).astype("int8")

# The two new columns are small integers
print(features[["type", "hour"]].dtypes.to_string())
# Hour, type and hour of day side by side
features[["step", "type", "hour"]].head()
""")

code("""
# Check your work
# type holds only 0 and 1
assert set(features["type"].unique()) == {0, 1}, "type should hold only 0 and 1"
# hour runs from 0 to 23
assert features["hour"].between(0, 23).all(), "hour should run from 0 to 23"
""")

md("""
The full table and the active slice are no longer needed, and they hold most of the memory. Delete them before
building features.
""")

code("""
# Free the memory held by the full table and the active slice
del df, active
# Confirm the memory was freed
print("df and active deleted")
""")

md("""
### Step 1.22: Mark the unrecorded balances

Set the unrecorded recipient balances (Step 1.19) to -1, a value that cannot occur naturally. Set the
unrecorded sender balances to `NaN`, "unknown".

**Concept: a marker value.** Filling a missing value with an average would hide that it was missing. A value
that cannot happen naturally (-1 for a balance) keeps that fact visible to a tree model, and tree models in
XGBoost handle `NaN` directly.
""")

code("""
# Recipient balances 0 -> 0 become -1
features.loc[recipient_zero_pair, ["oldBalanceDest", "newBalanceDest"]] = -1

# Sender balances 0 -> 0 become NaN
features.loc[sender_zero_pair, ["oldBalanceOrig", "newBalanceOrig"]] = np.nan

# Print how many balances were marked
print(f"recipient balances marked -1: {int(recipient_zero_pair.sum()):,}   sender balances set to NaN: {int(sender_zero_pair.sum()):,}")
""")

code("""
# Check your work
# 5,776 recipient pairs marked -1
assert int(recipient_zero_pair.sum()) == 5_776, f"expected 5,776 recipient markers, got {int(recipient_zero_pair.sum()):,}"
# 1,308,566 sender pairs set to NaN
assert int(sender_zero_pair.sum()) == 1_308_566, f"expected 1,308,566 sender NaNs, got {int(sender_zero_pair.sum()):,}"
""")

md("""
### Step 1.23: Compute the two ledger errors

Write the ledger rule as two numbers that should be zero for an honest transaction:

- `errorBalanceOrig` = new sender balance + amount - old sender balance
- `errorBalanceDest` = old recipient balance + amount - new recipient balance

**Concept: feature engineering.** Turning domain knowledge, here an accounting identity, into a new column is
called feature engineering. It often helps a model more than a bigger model does.
""")

code("""
# Sender side: how far the sender's balance change misses the amount
features["errorBalanceOrig"] = features["newBalanceOrig"] + features["amount"] - features["oldBalanceOrig"]

# Recipient side: how far the recipient's balance change misses the amount
features["errorBalanceDest"] = features["oldBalanceDest"] + features["amount"] - features["newBalanceDest"]

# Read them off rows 2 and 3
features.loc[[2, 3], ["amount", "errorBalanceOrig", "errorBalanceDest"]]
""")

code("""
# Check your work
# Row 2: 181 never arrived
assert features.loc[2, "errorBalanceDest"] == 181, "row 2: 181 should be missing on the recipient side"
# Row 3: 21,363 missing on the recipient side
assert features.loc[3, "errorBalanceDest"] == 21_363, "row 3: 21,182 + 181 should be missing on the recipient side"
""")

md("""
For the two frauds the sender-side error is 0, but the recipient side is off: by 181 for row 2 (the money
arrived nowhere; the -1 marker cancels out in the subtraction) and by 21,363 for row 3 (21,182 + 181 should have
been there, and 0 is).

### Step 1.24: Check whether the pattern holds across all transfers

Compare the recipient error for fraud and honest transfers, next to the raw amount.

**Concept: a sample for plotting.** Drawing 2.7 million points is slow and unreadable, so the chart uses 200,000
random honest rows plus every fraud.
""")

code("""
# 200,000 random rows plus every fraud
sample = features.sample(200_000, random_state=SEED).index.union(features.index[target == 1])
# Type and size of the sample index
print(type(sample), len(sample))
""")

code("""
# A blank figure with two histogram panels
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
# For each panel: its axis, the column and the title
panels = [(axes[0], "amount", "Amount"), (axes[1], "errorBalanceDest", "Recipient balance error")]
# Draw each panel's honest and fraud histograms
for panel_ax, panel_col, panel_title in panels:
    # Clip extreme values so the bins stay readable
    values = features.loc[sample, panel_col].clip(lower=-1e6, upper=1e7)
    panel_ax.hist(values[target[sample] == 0], bins=80, alpha=0.6, density=True, label="honest (sample)")
    panel_ax.hist(values[target[sample] == 1], bins=80, alpha=0.6, density=True, label="fraud (all)")
    panel_ax.set_title(panel_title)
    panel_ax.set_yscale("log")
    panel_ax.legend()
# Keep the two histogram panels apart
plt.tight_layout()
# Save them as figs/error_feature_distribution.png
plt.savefig("figs/error_feature_distribution.png", dpi=150, bbox_inches="tight")
# Show the two histograms
plt.show()
""")

code("""
# Share of frauds whose sender error is exactly zero
sender_exact_fraud = (features.loc[target == 1, "errorBalanceOrig"].abs() < 1e-6).mean()
# The same share for honest transfers
sender_exact_honest = (features.loc[target == 0, "errorBalanceOrig"].abs() < 1e-6).mean()
# Share of frauds whose recipient error is exactly zero
recipient_exact_fraud = (features.loc[target == 1, "errorBalanceDest"].abs() < 1e-6).mean()
# The recipient share for honest transfers
recipient_exact_honest = (features.loc[target == 0, "errorBalanceDest"].abs() < 1e-6).mean()

# Print the sender shares
print(f"sender error exactly 0:    fraud {sender_exact_fraud:.1%}, honest {sender_exact_honest:.1%}")
# Print the recipient shares
print(f"recipient error exactly 0: fraud {recipient_exact_fraud:.1%}, honest {recipient_exact_honest:.1%}")
""")

code("""
# Check your work
# 99.5% of frauds and 9.2% of honest transfers have an exact sender side
assert round(sender_exact_fraud * 100, 1) == 99.5 and round(sender_exact_honest * 100, 1) == 9.2, "sender exact-zero shares changed"
""")

md("""
The amounts overlap; the recipient error does not. And for fraud the *sender's* side of the ledger is exact in
99.5% of cases, while for honest transfers it is exact in only 9.2%, mostly because of the unrecorded sender
balances in Step 1.19. So "the sender's ledger is clean and the recipient's is not" is the combination that marks
a fraud, and Step 2.4 shows a two-question tree finding exactly that.

### Step 1.25: Compare the balance fingerprints of fraud and honest transfers

**Concept: correlation.** A correlation runs from -1 to 1 and says how closely two columns move together. In an
honest transfer the amount should match the balance change on both sides; in PaySim, fraud breaks that on the
recipient's side.
""")

code("""
# The balance block: the amount, the four balances and the two errors
block = ["amount", "oldBalanceOrig", "newBalanceOrig", "oldBalanceDest", "newBalanceDest", "errorBalanceOrig", "errorBalanceDest"]

# Correlation table for honest transfers
honest_corr = features.loc[target == 0, block].corr()
# Correlation table for frauds
fraud_corr = features.loc[target == 1, block].corr()
# Type and size of one table
print(type(honest_corr), honest_corr.shape)
""")

code("""
# A blank figure with two heatmap panels
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# Pair each panel with its label and table
heatmaps = [(axes[0], "honest", honest_corr), (axes[1], "fraud", fraud_corr)]
# Draw one annotated heatmap per panel
for heat_ax, heat_label, heat_table in heatmaps:
    sns.heatmap(heat_table, vmin=-1, vmax=1, cmap="RdBu_r", annot=True, fmt=".2f", square=True, cbar=False, ax=heat_ax)
    heat_ax.set_title(f"Correlation of balance fields, {heat_label} transactions")
# Keep the two heatmaps apart
plt.tight_layout()
# Save them as figs/balance_fingerprint.png, the chart the README shows
plt.savefig("figs/balance_fingerprint.png", dpi=150, bbox_inches="tight")
# Show the two heatmaps
plt.show()
""")

code("""
# The four correlations the text below quotes
fingerprint = pd.DataFrame({
    "honest": honest_corr.loc["amount", ["errorBalanceOrig", "newBalanceDest", "oldBalanceOrig", "errorBalanceDest"]],
    "fraud": fraud_corr.loc["amount", ["errorBalanceOrig", "newBalanceDest", "oldBalanceOrig", "errorBalanceDest"]],
})
# Show the four, rounded as quoted
fingerprint.round(2)
""")

code("""
# Check your work
# Honest transfers: the sender error moves with the amount (0.99)
assert round(fingerprint.loc["errorBalanceOrig", "honest"], 2) == 0.99, "honest amount-to-sender-error correlation changed"
# Frauds: it barely does (0.09)
assert round(fingerprint.loc["errorBalanceOrig", "fraud"], 2) == 0.09, "fraud amount-to-sender-error correlation changed"
# Frauds: the amount moves with the sender's old balance (0.86) and the recipient error (0.65)
assert round(fingerprint.loc["oldBalanceOrig", "fraud"], 2) == 0.86 and round(fingerprint.loc["errorBalanceDest", "fraud"], 2) == 0.65, "fraud fingerprint changed"
""")

md("""
Read the `amount` row of each heatmap (printed above). In honest transfers the sender error moves almost exactly
with the amount (0.99), because so many honest sender balances are unrecorded; in fraud it barely does (0.09),
because fraud debits the sender exactly. In fraud the amount moves with the sender's old balance (0.86), because a fraud
takes what the account holds, and with the recipient error (0.65), because what was taken does not arrive.

### Step 1.26: Fix the model inputs and run the leakage guard

`isFlaggedFraud` is PaySim's verdict on the transaction, `step` identifies the hour and so the period, and the
account IDs are unique labels. None of them is something a model should learn from.

**Concept: leakage.** *Leakage* is when a model sees information it would not have at prediction time. A leaked
column does not break a model, it improves the score, so nothing downstream would flag it. That is why the guard
is code that runs, not a sentence.
""")

code("""
# Columns that must never be model inputs
LEAKY_COLUMNS = ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest", "step"]

# An empty list for the model inputs
MODEL_COLS = []
# Add every column except step
for column in features.columns:
    if column != "step":
        MODEL_COLS.append(column)

# An empty list for any leaky column
leaked = []
# Collect any input that is on the leaky list
for column in MODEL_COLS:
    if column in LEAKY_COLUMNS:
        leaked.append(column)
# Print the inputs
print("model inputs:", MODEL_COLS)
# Print any leaked column (none expected)
print("leaked columns:", leaked)
""")

code("""
# Check your work
# No leaky column reached the inputs
assert not leaked, f"leaked columns reached the model inputs: {leaked}"
# Nine inputs
assert len(MODEL_COLS) == 9, f"expected 9 model inputs, got {len(MODEL_COLS)}"
""")

md("""
If `step` had been left in `MODEL_COLS`, this Check cell would stop the run with
`AssertionError: leaked columns reached the model inputs: ['step']`. The fix is to drop the column from the
inputs, not to delete the assert.

### Step 1.27: Split by time: train on the past, test on the future

**Concept: a time split.** A model is always used on transactions that happen *after* it was trained. A random
split would let it train on hour 700 and be tested on hour 300, and score better than it ever could in use.

```
hour 1 ............................ 490 | 491 ................ 743
              training (fit the model)  |  test (score it once)
```
""")

code("""
# True for every row at or before the split hour
train_mask = features["step"] <= SPLIT_STEP

# Training inputs and labels
X_train, y_train = features.loc[train_mask, MODEL_COLS], target[train_mask]

# Test inputs and labels
X_test, y_test = features.loc[~train_mask, MODEL_COLS], target[~train_mask]

# Types and sizes of the two input tables
print(type(X_train), X_train.shape, X_test.shape)
# Print the training rows, fraud rate and frauds
print(f"training: {len(X_train):>9,} transactions, fraud rate {y_train.mean():.2%}, {int(y_train.sum()):,} frauds")
# Print the same for the test window
print(f"test:     {len(X_test):>9,} transactions, fraud rate {y_test.mean():.2%}, {int(y_test.sum()):,} frauds")
""")

code("""
# Check your work
# Every training row comes before every test row
assert features.loc[train_mask, "step"].max() < features.loc[~train_mask, "step"].min(), "a training row is later than a test row"
# No row is lost
assert len(X_train) + len(X_test) == len(features), "rows were lost in the split"
# The test window holds 132,136 rows and 2,754 frauds
assert len(X_test) == 132_136 and int(y_test.sum()) == 2_754, "the test window should hold 132,136 rows and 2,754 frauds"
""")

md("""
The test window has about ten times the fraud rate of the training window (2.08% against 0.21%), for the reason
Step 1.15 showed: fraud continues at the same pace while honest volume falls away. Step 2.15 works out what that
does to precision, and Step 2.20 checks whether a model trained on the busy period holds up in the quiet one.
""")

# =====================================================================
# PART 2
# =====================================================================
md("""
---
# Part 2: Build Model

Part 2 has three stages. **Baseline** sets the numbers to beat: a documented rule and a two-question tree.
**Iterate** adds XGBoost, chooses a cut-off and tries three other model families. **Evaluate** checks the result
period by period, tests a resampling alternative, and explains single decisions.

## 2A. Baseline

### Step 2.1: Write one scoring function for every model

**Concept: PR-AUC.** Most models output a *score* between 0 and 1, and the cut-off that turns it into a flag is
a choice. A *precision-recall curve* shows precision and recall at every cut-off at once, and *PR-AUC* (average
precision, as scikit-learn's `average_precision_score` computes it) summarises the curve in one number. Random
guessing scores about the fraud rate of the test set, here about 0.02; 1.0 is perfect. *ROC-AUC* is the chance a
random fraud scores above a random honest transaction; it looks high even for weak fraud models, so it is
reported but never used to choose.
""")

code("""
# Every model's scores go into this dictionary, one entry per model
results = {}


# A function that scores one model and stores the result under its name
def score_model(name, scores):
    # Precision and recall at every possible cut-off
    precision, recall, _ = precision_recall_curve(y_test, scores)
    # The last precision/recall pair has no cut-off, so leave it out
    meets_floor = precision[:-1] >= 0.99
    # The best recall available while keeping precision at 99% or more
    if meets_floor.any():
        recall_at_99 = recall[:-1][meets_floor].max()
    else:
        recall_at_99 = 0.0
    # Store the three measures and print a one-line summary
    results[name] = {"PR-AUC": average_precision_score(y_test, scores),
                     "ROC-AUC": roc_auc_score(y_test, scores),
                     "Recall at 99% precision": recall_at_99}
    print(f"{name:34s} PR-AUC {results[name]['PR-AUC']:.4f}   recall at 99% precision {recall_at_99:.4f}")


# Confirm the function exists and nothing is scored yet
print(type(score_model), "| models scored so far:", len(results))
""")

code("""
# Check your work
# The function exists
assert callable(score_model), "score_model is not defined"
# Nothing is on the scoreboard yet
assert len(results) == 0, f"expected an empty scoreboard, got {len(results)} entries"
""")

md("""
### Step 2.2: Score the documented rule as the baseline

PaySim documents its flag as a rule on large transfers. Apply the simplest version directly: flag every TRANSFER
above 200,000.

**Concept: what is a baseline?** The simplest method anyone could use without a model. A model is only worth
building if it beats the baseline by a margin that matters.
""")

code("""
# 1 for a TRANSFER (type 0) above 200,000, else 0
rule = ((X_test["type"] == 0) & (X_test["amount"] > 200_000)).astype(int)
# Type and size of the rule's flags, and how many it flags
print(type(rule), rule.shape, "| flagged:", int(rule.sum()))

# Score the rule's flags as the first entry on the scoreboard
score_model("Baseline (200,000 TRANSFER rule)", rule)
""")

code("""
# Check your work
# The rule's PR-AUC is 0.0283
assert round(results["Baseline (200,000 TRANSFER rule)"]["PR-AUC"], 4) == 0.0283, "baseline PR-AUC changed"
""")

md("""
The rule scores a PR-AUC of 0.0283, barely above random guessing at about 0.02. That is the number to beat.

### Step 2.3: Train a tree that may ask only two questions

Before reaching for a powerful model, train one small enough to read: a decision tree of depth 2, using only the
two ledger errors.

**Concept: class weight.** `class_weight="balanced"` makes each rare fraud count as much in training as the many
honest rows together. Without it, the tree would do best by calling everything honest, like the lazy detector in
Step 1.13.
""")

code("""
# The decision tree, and a function that prints its rules
from sklearn.tree import DecisionTreeClassifier, export_text

# The two inputs the tree may use
tree_cols = ["errorBalanceOrig", "errorBalanceDest"]

# A depth-2 tree with balanced class weights
tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=SEED)

# Fit it; the tree cannot take NaN, so unknown sender errors are treated as 0
tree.fit(X_train[tree_cols].fillna(0), y_train)
# Type, depth and number of leaves
print(type(tree), "| depth:", tree.get_depth(), "| leaves:", tree.get_n_leaves())
""")

code("""
# Check your work
# Depth 2: two questions
assert tree.get_depth() == 2, "the tree should have depth 2"
""")

md("""
### Step 2.4: Read the tree's rules

Print the tree as if-then rules and compare them with the ledger reasoning of Step 1.9.

**Concept: a readable model.** A depth-2 tree is a short set of rules a fraud analyst can check by hand.
""")

code("""
# The tree's rules as text
tree_rules = export_text(tree, feature_names=tree_cols)
# Print them
print(tree_rules)
""")

code("""
# Check your work
# The tree splits on the recipient error at 61.52
assert "errorBalanceDest >  61.52" in tree_rules, "the recipient-error split changed"
# and on the sender error at 0.01
assert "errorBalanceOrig <= 0.01" in tree_rules, "the sender-error split changed"
""")

md("""
Read from the top. The tree calls a transaction fraud when the recipient's balance is short by more than about 62
(`errorBalanceDest > 61.52`) *and* the sender's balance moved by exactly the amount (`errorBalanceOrig <= 0.01`).
In words: the sender's side is clean, but the money never properly arrived. That is rows 2 and 3.

### Step 2.5: Count the tree's hits and misses on the test window

**Concept: confusion matrix.** A 2 x 2 count of every combination of truth and prediction: honest flagged as
honest, honest flagged as fraud (false alarms), fraud missed, and fraud caught.
""")

code("""
# A function that draws the confusion matrix and prints recall and precision
def show_confusion(truth, flags, title):
    # Count the four combinations of truth and prediction
    cm = confusion_matrix(truth, flags)
    # Draw them as a labelled heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["flagged honest", "flagged fraud"], yticklabels=["actually honest", "actually fraud"])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    # Recall and precision read off the matrix
    print(f"recall:    {cm[1, 1] / cm[1].sum():.1%}   precision: {cm[1, 1] / max(cm[:, 1].sum(), 1):.1%}")
    return cm


# The tree's flags on the test window
tree_flags = tree.predict(X_test[tree_cols].fillna(0))
# Draw the tree's matrix and keep the counts
tree_cm = show_confusion(y_test, tree_flags, "Two-question tree, test window")
# Type and size of the counts
print(type(tree_cm), tree_cm.shape)
""")

code("""
# Check your work
# 1,367 frauds caught
assert tree_cm[1, 1] == 1_367, f"expected the tree to catch 1,367 frauds, got {tree_cm[1, 1]:,}"
# 79 false alarms
assert tree_cm[0, 1] == 79, f"expected 79 false alarms, got {tree_cm[0, 1]}"
""")

md("""
Two questions catch 49.6% of the test fraud (1,367 of 2,754) with 94.5% precision (79 false alarms). That is
already far beyond the built-in flag. The frauds it misses need more than two questions.

### Step 2.6: Score the tree with the same function as every other model

Turn the tree's leaves into probabilities and put them through `score_model`.

**Concept: one yardstick.** Every model is scored by the same function on the same test rows, which is what makes
the scoreboard in Step 2.19 fair.
""")

code("""
# The tree's probability of fraud for each test row
tree_scores = tree.predict_proba(X_test[tree_cols].fillna(0))[:, 1]
# Type and length of the tree's scores
print(type(tree_scores), tree_scores.shape)

# Put the tree on the scoreboard
score_model("Two-question tree", tree_scores)
""")

code("""
# Check your work
# The tree's PR-AUC is 0.4892
assert round(results["Two-question tree"]["PR-AUC"], 4) == 0.4892, "tree PR-AUC changed"
""")

md("""
> **Try it:** change `max_depth=2` to `max_depth=4` in Step 2.3 and re-run Steps 2.3 to 2.6. How do recall and
> precision move, and how much harder are the rules to read?

## 2B. Iterate

### Step 2.7: Work out the class weight for XGBoost

XGBoost (Chen and Guestrin, 2016) builds many small trees, each one correcting the mistakes of the trees before
it, and uses all nine inputs.

**Concept: `scale_pos_weight`.** XGBoost's class weight. Set it to the number of honest training rows per fraud,
so each fraud counts as much as that many honest rows. Take it from the training labels only; using the test
labels would leak the answer into the model.
""")

code("""
# Honest training rows divided by fraud training rows
class_weight = (y_train == 0).sum() / (y_train == 1).sum()
# The weight is a single number
print(type(class_weight))
# Print the honest rows per fraud
print(f"honest transactions per fraud in training: {class_weight:.1f}")
""")

code("""
# Check your work
# 482.3 honest rows per fraud
assert round(class_weight, 1) == 482.3, f"expected 482.3, got {class_weight:.1f}"
""")

md("""
### Step 2.8: See the error a text column causes

A common first mistake is to hand a model a column of text. Try it on 1,000 rows with `type` turned back into
words, and read the error.

**Concept: data types.** Models do arithmetic, so every input must be a number (or a declared category).
""")

code("""
# XGBoost's scikit-learn style classifier
from xgboost import XGBClassifier

# 1,000 training rows with type written as text again
text_rows = X_train.head(1_000).copy()
# Write the type back as words
text_rows["type"] = text_rows["type"].map({0: "TRANSFER", 1: "CASH_OUT"})
# Confirm the column now holds text
print(text_rows["type"].dtype)

# Try to fit, and keep the error message instead of stopping the notebook
try:
    XGBClassifier(n_estimators=2, n_jobs=N_JOBS).fit(text_rows, y_train.head(1_000))
    text_column_error = None
except (ValueError, TypeError) as error:
    text_column_error = str(error)
# Print the start of the error message
print(str(text_column_error)[:300])
""")

code("""
# Check your work
# Fitting on text raised an error
assert text_column_error is not None, "fitting on a text column should raise an error"
# Free the text copy
del text_rows
""")

md("""
XGBoost refuses a column of strings and names the column it cannot use. The fix is the one Step 1.21 already
made: map the type to 0 and 1.

### Step 2.9: Train XGBoost

Shallow trees (depth 3), library defaults otherwise, and the class weight from Step 2.7.

**Concept: gradient boosting.** Each new tree is fitted to the mistakes the trees before it still make, so the
model improves in many small steps.
""")

code("""
# A gradient-boosted model with shallow trees and the training class weight
model = XGBClassifier(max_depth=3, scale_pos_weight=class_weight, n_jobs=N_JOBS)

# Fit it on the training window
model.fit(X_train, y_train)
# Type and number of trees
print(type(model), "| trees:", model.get_booster().num_boosted_rounds())
""")

code("""
# Check your work
# XGBoost's default of 100 trees
assert model.get_booster().num_boosted_rounds() == 100, "expected 100 trees"
""")

md("""
### Step 2.10: Score XGBoost on the test window

Score the test rows and put XGBoost on the scoreboard.

**Concept: scoring on unseen rows.** The test window was never used in training, so its PR-AUC estimates how the
model does on later transactions of the same simulator.
""")

code("""
# The model's fraud score for each test row
xgb_scores = model.predict_proba(X_test)[:, 1]
# Type and length of XGBoost's scores
print(type(xgb_scores), xgb_scores.shape)

# Put XGBoost on the scoreboard
score_model("XGBoost, class-weighted", xgb_scores)
""")

code("""
# Check your work
# XGBoost's PR-AUC is 0.9989
assert round(results["XGBoost, class-weighted"]["PR-AUC"], 4) == 0.9989, "XGBoost PR-AUC changed"
""")

md("""
PR-AUC rises from 0.4892 for the two-question tree to 0.9989, and the model keeps 99% precision while catching
99.89% of the frauds. Step 2.19 asks whether a score this high should be believed.

### Step 2.11: Draw the precision-recall and ROC curves

**Concept: why the PR curve matters more here.** Honest transactions are so many that a small false-alarm *rate*
still means many false alarms. The ROC curve hides that; the precision-recall curve shows it.
""")

code("""
# A blank figure: precision-recall on the left, ROC on the right
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# The two models to draw, each with its scores and colour
curves = [("Two-question tree", tree_scores, "#7f8c8d"), ("XGBoost", xgb_scores, "#2e5c8a")]
# Draw each model's precision-recall and ROC curves
for curve_name, curve_scores, curve_color in curves:
    # Precision-recall curve
    curve_precision, curve_recall, _ = precision_recall_curve(y_test, curve_scores)
    axes[0].plot(curve_recall, curve_precision, color=curve_color, lw=2,
                 label=f"{curve_name} (PR-AUC {average_precision_score(y_test, curve_scores):.3f})")
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, curve_scores)
    axes[1].plot(fpr, tpr, color=curve_color, lw=2, label=f"{curve_name} (ROC-AUC {roc_auc_score(y_test, curve_scores):.3f})")

# Random guessing on the PR chart: a flat line at the fraud rate
axes[0].axhline(y_test.mean(), color="#a0aec0", ls="--", lw=1, label=f"random ({y_test.mean():.3f})")
# Label the PR chart's x-axis
axes[0].set_xlabel("Recall (share of frauds caught)")
# Label the PR chart's y-axis
axes[0].set_ylabel("Precision (share of alarms that are fraud)")
# Title of the PR chart
axes[0].set_title("Precision-recall curve, test window")
# Key for the PR chart
axes[0].legend(loc="lower left")
# Random guessing on the ROC chart: the diagonal
axes[1].plot([0, 1], [0, 1], color="#a0aec0", ls="--", lw=1, label="random")
# Label the ROC chart's x-axis
axes[1].set_xlabel("False-alarm rate (share of honest transactions flagged)")
# Label the ROC chart's y-axis
axes[1].set_ylabel("Recall")
# Title of the ROC chart
axes[1].set_title("ROC curve, test window")
# Key for the ROC chart
axes[1].legend(loc="lower right")
# Keep the two curve panels apart
plt.tight_layout()
# Save them as figs/pr_roc_curves.png
plt.savefig("figs/pr_roc_curves.png", dpi=150, bbox_inches="tight")
# Show the curves
plt.show()
""")

code("""
# Check your work
# The curves were saved
assert os.path.isfile("figs/pr_roc_curves.png"), "figs/pr_roc_curves.png was not written"
""")

md("""
XGBoost's precision-recall curve hugs the top-right corner: it catches nearly all the fraud before precision
falls. The tree's curve drops at half recall, where its two questions run out. Both ROC curves look almost
perfect, which is why ROC-AUC is not used to choose anything.

### Step 2.12: Find the cut-off that keeps precision at 99%

A score is not a decision. One common policy: catch as much fraud as possible while at least 99 of every 100
alarms are real.

**Concept: operating threshold.** The cut-off a model runs at in use. Above it a transfer is frozen; below it,
it goes through.
""")

code("""
# Precision, recall and the cut-off behind each pair
precision, recall, thresholds = precision_recall_curve(y_test, xgb_scores)

# Cut-offs that keep precision at 99% or more
eligible = precision[:-1] >= 0.99

# Among those, the one with the highest recall
best = int(eligible.nonzero()[0][recall[:-1][eligible].argmax()])
# The cut-off at that position
operating_threshold = float(thresholds[best])
# Print the cut-off with its precision and recall
print(f"cut-off: {operating_threshold:.4f}   precision {precision[best]:.4f}   recall {recall[best]:.4f}")
""")

code("""
# Check your work
# The cut-off is 0.3602
assert round(operating_threshold, 4) == 0.3602, f"expected a cut-off of 0.3602, got {operating_threshold:.4f}"
""")

md("""
### Step 2.13: Count hits and misses at that cut-off

Flag every test row at or above the cut-off and count the four outcomes.

**Concept: false alarms as a count.** Precision is a ratio; the number of honest customers frozen is the work the
fraud team actually has to handle.
""")

code("""
# Flag every test row at or above the cut-off
flags = (xgb_scores >= operating_threshold).astype(int)

# Confusion matrix at the cut-off
cm = show_confusion(y_test, flags, f"XGBoost at the 99%-precision cut-off ({operating_threshold:.2f})")
# Type of the counts, and the four counts in a row
print(type(cm), cm.ravel())
""")

code("""
# Check your work
# 3 missed and 27 false alarms
assert cm[1, 0] == 3 and cm[0, 1] == 27, f"expected 3 missed and 27 false alarms, got {cm[1, 0]} and {cm[0, 1]}"
""")

md("""
At this cut-off the model misses 3 frauds and raises 27 false alarms among the 129,382 honest test transactions.

### Step 2.14: Find the cheapest cut-off at a 10-to-1 cost

A second policy weighs the two mistakes directly. Price a missed fraud at 10 units (the money is lost) and a false
alarm at 1 unit (a review and a customer locked out of their money), so `COST_RATIO = 10`. These prices are an
assumption made for teaching, not costs measured at any operator. Try 99 cut-offs and keep the cheapest.

**Concept: cost-sensitive threshold.** Cost = 10 x missed frauds + false alarms. The cheapest cut-off depends on
the ratio, and the ratio is a business input, not something the data can tell you.
""")

code("""
# 99 cut-offs from 0.01 to 0.99
cutoffs = np.linspace(0.01, 0.99, 99)

# An empty list for the missed frauds at each cut-off
missed_list = []
# An empty list for the false alarms
false_alarm_list = []
# Count both at each cut-off
for cutoff in cutoffs:
    missed_list.append(int(((xgb_scores < cutoff) & (y_test == 1)).sum()))
    false_alarm_list.append(int(((xgb_scores >= cutoff) & (y_test == 0)).sum()))
# The missed counts as an array
missed = np.array(missed_list)
# The false-alarm counts as an array
false_alarms = np.array(false_alarm_list)
# Type and lengths of the two arrays
print(type(missed), missed.shape, false_alarms.shape)
""")

code("""
# Cost of each cut-off: COST_RATIO units per missed fraud, 1 unit per false alarm
cost = COST_RATIO * missed + false_alarms
# Position of the cheapest
cheapest = int(cost.argmin())
# Print the cheapest cut-off and its counts
print(f"cheapest cut-off {cutoffs[cheapest]:.2f}: {missed[cheapest]} missed, {false_alarms[cheapest]} false alarms")
""")

code("""
# Check your work
# The cheapest cut-off is 0.96
assert round(cutoffs[cheapest], 2) == 0.96, f"expected 0.96, got {cutoffs[cheapest]:.2f}"
# It misses 3 and raises 2 false alarms
assert missed[cheapest] == 3 and false_alarms[cheapest] == 2, "the cheapest cut-off's counts changed"
""")

code("""
# A blank chart for the cost curve
fig, ax = plt.subplots(figsize=(10, 4.5))
# The cost at each cut-off
ax.plot(cutoffs, cost, color="#2e5c8a")
# A dashed line at the cheapest cut-off
ax.axvline(cutoffs[cheapest], color="#c0504d", linestyle="--", label=f"cheapest cut-off {cutoffs[cheapest]:.2f}")
# A dotted line at the 99%-precision cut-off
ax.axvline(operating_threshold, color="gray", linestyle=":", label=f"99%-precision cut-off {operating_threshold:.2f}")
# Label the cost chart's x-axis
ax.set_xlabel("Cut-off")
# Label its y-axis with the cost formula
ax.set_ylabel(f"Cost = {COST_RATIO} x missed + false alarms")
# Title of the cost chart
ax.set_title("Cost of each cut-off on the test window")
# Key for the two lines
ax.legend()
# Keep the cost chart inside its margins
plt.tight_layout()
# Save it as figs/cost_curve.png
plt.savefig("figs/cost_curve.png", dpi=150, bbox_inches="tight")
# Show the cost curve
plt.show()
""")

md("""
The two policies pick different cut-offs, 0.36 and 0.96, and neither is "correct": the cost ratio has to come
from the business that will run the model. (The shipped pipeline prices a missed fraud at 1,000 units and a false
alarm at 10, `COST_PER_MISSED_FRAUD` and `COST_PER_FALSE_ALARM` in `src/features.py`: also an assumption, a ratio
of 100 to 1.)

> **Try it:** set `COST_RATIO = 100` in Step 1.2, re-run it and Step 2.14. Where does the cheapest cut-off move,
> and why?

### Step 2.15: Restate precision at the real fraud rate

The test window is fraud-heavy: 2.08% of its transactions are fraud, against about 0.30% of all TRANSFER and
CASH_OUT transactions over the month. Precision depends on that mix.

**Concept: precision depends on the base rate.** The model's two rates (the share of frauds it catches and the
share of honest transactions it wrongly flags) are properties of the model. Precision also depends on how many
honest transactions it meets per fraud, so a rarer fraud mix means lower precision from the same model.
""")

code("""
# Share of test frauds caught at the 99%-precision cut-off
catch_rate = cm[1, 1] / cm[1].sum()
# Share of honest test transactions wrongly flagged
false_alarm_rate = cm[0, 1] / cm[0].sum()

# TRANSFER and CASH_OUT fraud rate over the whole month
real_fraud_rate = target.mean()
# Print the two rates and the real fraud rate
print(f"catch rate {catch_rate:.4f}, false-alarm rate {false_alarm_rate:.6f}, real fraud rate {real_fraud_rate:.4%}")
""")

code("""
# Frauds in a million transactions at the real fraud rate
frauds_per_million = 1_000_000 * real_fraud_rate
# Frauds caught among them
caught_per_million = catch_rate * frauds_per_million
# False alarms among the honest ones
false_alarms_per_million = false_alarm_rate * (1_000_000 - frauds_per_million)
# Precision at that mix
precision_at_real_rate = caught_per_million / (caught_per_million + false_alarms_per_million)

# Print the counts per million
print(f"per million transactions: {frauds_per_million:,.0f} frauds, {caught_per_million:,.0f} caught, {false_alarms_per_million:,.0f} false alarms")
# Print the precision, beside the test-window figure
print(f"precision at the real fraud rate: {precision_at_real_rate:.1%}  (on the test window: {precision[best]:.1%})")
""")

code("""
# Check your work
# 93.4% precision at the real fraud rate
assert round(precision_at_real_rate * 100, 1) == 93.4, f"expected 93.4%, got {precision_at_real_rate:.1%}"
""")

md("""
At the whole-month fraud rate, about 208 false alarms per million transactions sit beside 2,961 caught frauds,
so precision falls to 93.4%: roughly one alarm in fifteen is false, not one in a hundred. Nothing about the model
changed; only the mix did. A fraud team sizing its review queue needs this number. The shipped pipeline reports
the same calculation for its own model (`precision_at_real_fraud_rate` in `dashboard/data/business_impact.json`).

### Step 2.16: Try a linear model: logistic regression

Is XGBoost special, or would other model families do as well? Start with the simplest.

**Concept: pipeline.** A scikit-learn `Pipeline` chains preparation and model, so the same steps run in the same
order on training and test data. Logistic regression cannot take `NaN`, so the unknown sender balances are filled
with 1 (a value that cannot be confused with a real balance or with the -1 marker), and each input is rescaled.
""")

code("""
# Fill-in for missing values
from sklearn.impute import SimpleImputer
# Logistic regression
from sklearn.linear_model import LogisticRegression
# Chains the steps into one model
from sklearn.pipeline import Pipeline
# Rescales each input to mean 0 and spread 1
from sklearn.preprocessing import StandardScaler

# Fill missing values with 1, rescale, then fit a class-weighted logistic regression
logistic = Pipeline([
    ("fill", SimpleImputer(strategy="constant", fill_value=1)),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])
# Type and the names of the three steps
print(type(logistic), list(logistic.named_steps))
""")

code("""
# Fit the logistic regression on the training window
logistic.fit(X_train, y_train)
# Put it on the scoreboard, scored on the test window
score_model("Logistic regression", logistic.predict_proba(X_test)[:, 1])
""")

code("""
# Check your work
# Logistic regression's PR-AUC is 0.7901
assert round(results["Logistic regression"]["PR-AUC"], 4) == 0.7901, "logistic regression PR-AUC changed"
""")

md("""
### Step 2.17: Try a random forest

**Concept: random forest** (Breiman, 2001). Hundreds of deep trees, each trained on a random sample of rows,
voting together.
""")

code("""
# The random forest model
from sklearn.ensemble import RandomForestClassifier

# 200 trees, at most 16 levels deep, at least 20 rows per leaf
forest = RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=20,
                                class_weight="balanced_subsample", n_jobs=RF_JOBS, random_state=SEED)

# Fit the forest on the training window
forest.fit(X_train, y_train)
# Score the forest's test-window probabilities
score_model("Random forest", forest.predict_proba(X_test)[:, 1])
""")

code("""
# Check your work
# The forest's PR-AUC rounds to 1.0
assert round(results["Random forest"]["PR-AUC"], 4) == 1.0, "random forest PR-AUC changed"
""")

md("""
### Step 2.18: Try LightGBM

**Concept: LightGBM** (Ke et al., 2017). Another gradient-boosting library, which grows trees leaf by leaf. Here
it gets the same class weight, larger trees and very small leaves (`min_child_samples=5`).
""")

code("""
# The LightGBM model
from lightgbm import LGBMClassifier

# 400 trees of up to 31 leaves, at least 5 rows per leaf
lgbm = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=5,
                      scale_pos_weight=class_weight, n_jobs=N_JOBS, random_state=SEED, verbose=-1)

# Fit LightGBM on the training window
lgbm.fit(X_train, y_train)
# Score LightGBM's test-window probabilities
score_model("LightGBM", lgbm.predict_proba(X_test)[:, 1])
""")

code("""
# Check your work
# LightGBM's PR-AUC is 0.0338
assert round(results["LightGBM"]["PR-AUC"], 4) == 0.0338, "LightGBM PR-AUC changed"
""")

md("""
### Step 2.19: Put every model side by side

**Concept: a scoreboard.** One table, one metric to sort by (PR-AUC), every model scored on the same test rows by
the same function.
""")

code("""
# One row per model, sorted by PR-AUC
scoreboard = pd.DataFrame(results).T.sort_values("PR-AUC", ascending=False)
# Type and size of the scoreboard
print(type(scoreboard), scoreboard.shape)
# Print it with four decimals
print(scoreboard.to_string(float_format="{:.4f}".format))
""")

code("""
# PR-AUC from lowest to highest, for the chart
pr_auc_sorted = scoreboard["PR-AUC"].sort_values()

# An empty list of bar colours
bar_colours = []
# Red for a model below 0.5 PR-AUC, blue for the rest
for value in pr_auc_sorted:
    if value < 0.5:
        bar_colours.append("#c0504d")
    else:
        bar_colours.append("#2e5c8a")

# A blank chart for the scoreboard
fig, ax = plt.subplots(figsize=(9, 4.5))
# One horizontal bar per model
pr_auc_sorted.plot(kind="barh", ax=ax, color=bar_colours)
# Label the x-axis, with the random-guess level
ax.set_xlabel("PR-AUC on the test window (random guessing scores about 0.02)")
# Title of the scoreboard chart
ax.set_title("Model comparison on the test window")
# Room on the right for the value labels
ax.set_xlim(0, 1.05)
# Write each model's PR-AUC beside its bar
for bar_position, bar_value in enumerate(pr_auc_sorted):
    ax.text(bar_value + 0.01, bar_position, f"{bar_value:.4f}", va="center", fontsize=9)
# Keep the scoreboard chart inside its margins
plt.tight_layout()
# Save it as figs/model_comparison.png
plt.savefig("figs/model_comparison.png", dpi=150, bbox_inches="tight")
# Show the scoreboard chart
plt.show()
""")

code("""
# Check your work
# The random forest ranks first
assert scoreboard.index[0] == "Random forest", "the random forest should rank first"
# Six models so far
assert len(scoreboard) == 6, f"expected 6 models on the scoreboard, got {len(scoreboard)}"
""")

md("""
Three lessons are in this table.

1. **A perfect score is a warning, not a win.** The random forest reaches PR-AUC 1.0000. Real fraud is never this
   separable. PaySim generates fraud by a simple process, and once the ledger errors exist the label follows
   from them almost exactly. On a simulator, the scores show that the method works; they do not tell you how
   accurate a model would be on a real network.
2. **Models can fail quietly.** LightGBM falls to 0.0338, about the same as the 200,000 rule (0.0283), without any
   error message. This notebook does not test why; its settings (a class weight of 482 with leaves as small as 5
   rows) are the first thing to check. The point is that nothing warned: only scoring on later data showed it.
   Always do that before trusting a model.
3. **A linear model is not enough here.** Logistic regression reaches 0.7901 but keeps 99% precision only while
   catching 45% of the fraud, most likely because the fraud pattern ("this error is zero *and* that one is
   not") is a combination a linear model cannot express directly.

XGBoost stays the model for the rest of the notebook: it is within 0.0011 of the forest, and SHAP explains a
single boosted model quickly. On a simulator this separable, the gap between the two says nothing about which
would do better on real data.

## 2C. Evaluate

### Step 2.20: Score XGBoost period by period

A single test score hides change over time. Split the test window into four periods and score each, keeping the
cut-off from Step 2.12 fixed, as it would be in use.

**Concept: stability over time.** A model that is good on average but failing in the most recent period is
failing now.
""")

code("""
# The hour of each test row
test_steps = features.loc[~train_mask, "step"].to_numpy()

# An empty list, one row of measures per period
period_rows = []
# Score each of the four test periods
for start, end in [(491, 560), (561, 630), (631, 700), (701, 743)]:
    # Rows in this period, their labels and their scores
    in_period = (test_steps >= start) & (test_steps <= end)
    truth = y_test.to_numpy()[in_period]
    scores = xgb_scores[in_period]
    # Flags at the fixed cut-off, and the frauds among them
    flagged_period = scores >= operating_threshold
    true_positives = (flagged_period & (truth == 1)).sum()
    period_rows.append({"hours": f"{start}-{end}",
                        "transactions": int(in_period.sum()),
                        "frauds": int(truth.sum()),
                        "PR-AUC": average_precision_score(truth, scores),
                        "precision": true_positives / max(flagged_period.sum(), 1),
                        "recall": true_positives / truth.sum()})
# Type and number of periods scored
print(type(period_rows), len(period_rows))
""")

code("""
# Turn the rows into a table, one row per period
periods = pd.DataFrame(period_rows).set_index("hours")
# Print the period table with four decimals
print(periods.to_string(float_format="{:.4f}".format))
""")

code("""
# Check your work
# The four periods cover the whole test window
assert periods["transactions"].sum() == len(X_test), "the four periods should cover the whole test window"
# Recall stays above 99% in every period
assert periods["recall"].min() > 0.99, "recall fell below 99% in a period"
""")

md("""
### Step 2.21: Plot the three measures by period

Draw the period table as three lines.

**Concept: a fixed cut-off over time.** In use the cut-off does not move every week, so precision and recall at one
fixed cut-off show what customers and the fraud team would see.
""")

code("""
# A blank chart for the three measures
fig, ax = plt.subplots(figsize=(9, 4.5))
# One line per measure across the periods
periods[["PR-AUC", "precision", "recall"]].plot(marker="o", ax=ax)
# Zoom the y-axis on the top of the scale
ax.set_ylim(0.95, 1.005)
# Label the x-axis with the period
ax.set_xlabel("Test period (simulated hours)")
# Label the y-axis
ax.set_ylabel("Score")
# Title naming the fixed cut-off
ax.set_title(f"XGBoost by test period, cut-off fixed at {operating_threshold:.2f}")
# Key for the three lines
ax.legend(loc="lower left")
# Keep the period chart inside its margins
plt.tight_layout()
# Save it as figs/performance_over_time.png
plt.savefig("figs/performance_over_time.png", dpi=150, bbox_inches="tight")
# Show the period chart
plt.show()
""")

code("""
# Check your work
# The period chart was saved
assert os.path.isfile("figs/performance_over_time.png"), "figs/performance_over_time.png was not written"
# Precision at the fixed cut-off stays at 98.58% or more
assert round(periods["precision"].min() * 100, 2) == 98.58, f"lowest period precision changed: {periods['precision'].min():.4%}"
""")

md("""
Ranking (PR-AUC) and recall stay close to perfect in every period, and precision stays between 98.58% and 100% at
the fixed cut-off, even though the mix shifts sharply: the last period has 5,217 transactions against 50,526 in
the first, and 490 of them are fraud. On real data this check would run every week, and a fall in precision or
recall would trigger a review of the cut-off or a retrain. The shipped pipeline does a stricter version
(`src/validate.py` retrains on four expanding windows and tests each on the hours after it, which is called
walk-forward validation; `src/monitoring.py` tracks how the inputs drift with the Population Stability Index,
PSI, which compares how a feature's values spread across bins in a new window with the training window: below
0.10 is read as stable, 0.25 or more as a shift to investigate).

### Step 2.22: Fill the missing values for resampling

A common textbook alternative to class weights is SMOTE (Chawla et al., 2002): create synthetic frauds by
drawing points between real frauds until the classes are balanced. SMOTE cannot handle `NaN`, so fill it with -1
first.

**Concept: resampling.** Changing the rows a model trains on (here, adding synthetic frauds) instead of changing
how much each row counts (the class weight).
""")

code("""
# Fill NaN with -1 in copies of the training and test inputs
X_train_filled, X_test_filled = X_train.fillna(-1), X_test.fillna(-1)
# Type, size and NaN left in the filled training copy
print(type(X_train_filled), X_train_filled.shape, "| NaN left:", int(X_train_filled.isna().sum().sum()))
""")

code("""
# Check your work
# No NaN left in the training copy
assert int(X_train_filled.isna().sum().sum()) == 0, "the training copy still has NaN"
# Nor in the test copy
assert int(X_test_filled.isna().sum().sum()) == 0, "the test copy still has NaN"
""")

md("""
### Step 2.23: Create the synthetic frauds

Resample the training rows until the classes are equal.

**Concept: how SMOTE makes a row.** For each fraud it picks one of its five nearest fraud neighbours and creates a
new row at a random point on the line between them.
""")

code("""
# SMOTE, from the imbalanced-learn library
from imblearn.over_sampling import SMOTE

# Add synthetic frauds until there are as many frauds as honest rows
X_smote, y_smote = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(X_train_filled, y_train)
# Print the rows and frauds before and after
print(f"training rows: {len(X_train_filled):,} -> {len(X_smote):,}; frauds: {int(y_train.sum()):,} -> {int(y_smote.sum()):,}")
""")

code("""
# Check your work
# Equal frauds and honest rows after SMOTE
assert int(y_smote.sum()) == int((y_train == 0).sum()), "after SMOTE the classes should be equal"
""")

md("""
### Step 2.24: Train the same XGBoost on the resampled rows and score it

The classes are now balanced, so the class weight is 1.

**Concept: change one thing at a time.** The model settings are those of Step 2.9; only the training rows and the
class weight differ, so any difference in the score comes from the resampling.
""")

code("""
# Same settings as Step 2.9, but no class weight
smote_model = XGBClassifier(max_depth=3, scale_pos_weight=1, n_jobs=N_JOBS).fit(X_smote, y_smote)

# Put the SMOTE model on the scoreboard, scored on the filled test inputs
score_model("XGBoost + SMOTE", smote_model.predict_proba(X_test_filled)[:, 1])

# Free the resampled copies
del X_smote, y_smote, X_train_filled, X_test_filled
""")

code("""
# Check your work
# The SMOTE model's PR-AUC is 0.9995
assert round(results["XGBoost + SMOTE"]["PR-AUC"], 4) == 0.9995, "SMOTE PR-AUC changed"
""")

md("""
SMOTE scores 0.9995 against 0.9989 for class weighting, a difference of 0.06 points, and it doubled the training
data to get there. On this data the two approaches are equivalent, and class weights are the cheaper choice. (The
comparison also changed how missing values are handled, so it does not settle the question in general.)

### Step 2.25: Build a SHAP explainer for XGBoost

If a customer's money is frozen, someone has to say why.

**Concept: SHAP** (Lundberg and Lee, 2017). SHAP splits a model's score for one transaction into a contribution
from each input, measured against an average transaction. Positive values push towards fraud, negative towards
honest. For tree models `TreeExplainer` computes them exactly.
""")

code("""
# SHAP's progress bar warning is about the display, not the analysis
warnings.filterwarnings("ignore", message="IProgress not found")
# The SHAP library
import shap

# An explainer for the XGBoost model
explainer = shap.TreeExplainer(model)
# Type of the explainer
print(type(explainer))
""")

code("""
# Check your work
# A tree explainer
assert isinstance(explainer, shap.TreeExplainer), f"expected a TreeExplainer, got {type(explainer)}"
""")

md("""
### Step 2.26: Explain one fraud and one honest transfer

Explain the highest-scoring fraud and the first honest TRANSFER of the test window.

**Concept: a local explanation.** One transaction's SHAP values, added to the average, give that transaction's
score on the model's internal log-odds scale, so they say why this one was flagged.
""")

code("""
# The highest-scoring fraud in the test window
fraud_row = X_test[y_test == 1].iloc[[int(np.argmax(xgb_scores[(y_test == 1).to_numpy()]))]]
# The first honest TRANSFER in the test window
honest_row = X_test[(y_test == 0) & (X_test["type"] == 0)].iloc[[0]]
# XGBoost's score for the fraud
fraud_score = float(model.predict_proba(fraud_row)[0, 1])
# XGBoost's score for the honest transfer
honest_score = float(model.predict_proba(honest_row)[0, 1])
# For each of the two rows, print the score and the three inputs that pushed it most
for explain_label, explain_row in [("fraud", fraud_row), ("honest", honest_row)]:
    # Each input's push on this row's score
    contributions = pd.Series(explainer.shap_values(explain_row)[0], index=MODEL_COLS)
    # The three largest pushes, by size
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)[:3]
    print(f"{explain_label} transaction, score {model.predict_proba(explain_row)[0, 1]:.4f}; largest pushes:")
    for push_name, push_value in top.items():
        print(f"   {push_name:18s} value {explain_row[push_name].iloc[0]:>14,.2f}   push {push_value:+.2f}")
""")

code("""
# Check your work
# The fraud scores near 1
assert fraud_score > 0.99, f"expected the fraud to score above 0.99, got {fraud_score:.4f}"
# The honest transfer scores near 0
assert honest_score < 0.01, f"expected the honest transfer to score below 0.01, got {honest_score:.4f}"
""")

md("""
The fraud's three largest pushes are all towards fraud: the -1 marker on the recipient's new balance (the money
arrived nowhere), the sender's old balance, and a sender error of exactly 0. The honest transfer is pushed down
most by a large sender error: the sender sent more than its recorded balance, which PaySim's frauds almost never
do.

### Step 2.27: Explain 2,000 test transfers at once

**Concept: a global explanation.** Averaging the size of each input's SHAP values over many rows ranks the inputs
by how much they move the score overall.

The summary plot has one row per input and one dot per transaction, coloured by the input's value and placed by
how far it pushed the score.
""")

code("""
# 2,000 random test rows and their SHAP values
shap_sample = X_test.sample(2_000, random_state=SEED)
# SHAP values for those rows
shap_values = explainer.shap_values(shap_sample)
# Type and size: one value per row and input
print(type(shap_values), shap_values.shape)

# Draw the summary plot without showing it yet
shap.summary_plot(shap_values, shap_sample, show=False, max_display=9)
# Title of the summary plot
plt.title("What pushes the XGBoost score, 2,000 test transactions")
# Keep the summary plot inside its margins
plt.tight_layout()
# Save it as figs/shap_summary.png
plt.savefig("figs/shap_summary.png", dpi=150, bbox_inches="tight")
# Show the summary plot
plt.show()
""")

code("""
# Average size of each input's push: its overall importance
importance = pd.Series(np.abs(shap_values).mean(axis=0), index=MODEL_COLS).sort_values(ascending=False)
# Heading for the importance list
print("mean |SHAP| by input:")
# Print each input's mean |SHAP|, largest first
print(importance.round(3).to_string())
""")

code("""
# Check your work
# newBalanceOrig first, errorBalanceOrig second
assert importance.index[0] == "newBalanceOrig" and importance.index[1] == "errorBalanceOrig", "the top two inputs changed"
""")

md("""
The sender-side error is second, as the ledger reasoning predicted, but the raw sender balance after the transfer
(`newBalanceOrig`) matters most. That is a warning sign: almost every PaySim fraud empties the sender's account
(Step 1.19), so "the balance went to zero" works as a shortcut. It would also flag an honest customer closing
their own account. The shipped pipeline removes the raw balances for this reason (Step 3.6).
""")

# =====================================================================
# PART 3
# =====================================================================
md("""
---
# Part 3: Communicate Results

This part speaks to a fraud-operations manager. A fraud team needs two numbers from any screen: how much of the
fraud value it stops, and how many honest customers it freezes to do so. A fraud's value is its transaction
amount; a frozen honest customer is a false alarm. Both are counted on the test window, in PaySim's currency
units.

### Step 3.1: Total the fraud value in the test window

**Concept: value, not count.** Two screens that catch the same number of frauds can stop very different amounts
of money. A fraud team is judged on money.
""")

code("""
# Amounts and labels of the test rows, as arrays
amount_test = X_test["amount"].to_numpy()
# True where a test row is fraud
truth = y_test.to_numpy() == 1

# Total value of all test-window fraud
total_fraud_value = amount_test[truth].sum()
# The total is a single number
print(type(total_fraud_value))
# Print the test-window totals
print(f"test window: {len(y_test):,} transactions, {int(truth.sum()):,} frauds worth {total_fraud_value / 1e9:.2f} billion")
""")

code("""
# Check your work
# 4.19 billion of fraud in the test window
assert round(total_fraud_value / 1e9, 2) == 4.19, f"expected 4.19 billion, got {total_fraud_value / 1e9:.2f}"
""")

md("""
### Step 3.2: Write one function that counts what a screen stops and what it costs

**Concept: one function for every screen.** Every screen is counted by the same function, so the rows of the
Step 3.3 table differ only in the flags, never in the arithmetic.
""")

code("""
# A function that counts what a set of flags stops and what it freezes
def screen_outcome(flagged):
    # Make sure the flags are True/False
    flagged = np.asarray(flagged).astype(bool)
    # Fraud value the screen stopped
    stopped = amount_test[flagged & truth].sum()
    # Counts and values for frauds caught, frauds missed and honest customers frozen
    return {"frauds caught": int((flagged & truth).sum()), "frauds missed": int((~flagged & truth).sum()),
            "fraud value stopped": stopped, "share of fraud value stopped": stopped / total_fraud_value,
            "honest customers frozen": int((flagged & ~truth).sum()), "honest value frozen": amount_test[flagged & ~truth].sum()}


# Try it on a screen that flags nothing
nothing_outcome = screen_outcome(np.zeros(len(y_test), dtype=bool))
# Print the counts for the empty screen
print(nothing_outcome)
""")

code("""
# Check your work
# An empty screen stops no fraud
assert nothing_outcome["frauds caught"] == 0 and nothing_outcome["fraud value stopped"] == 0, "an empty screen should stop nothing"
# and freezes no honest customer
assert nothing_outcome["honest customers frozen"] == 0, "an empty screen should freeze no one"
""")

md("""
### Step 3.3: Count it for every screen seen so far

No screening, PaySim's built-in flag, the 200,000 rule, the two-question tree, and XGBoost at the two cut-offs
from Steps 2.12 and 2.14.

**Concept: the business-impact table.** One row per screen: frauds caught and missed, fraud value stopped, and
honest customers frozen with the value of their transfers.
""")

code("""
# Each screen's flags on the test window
screens = {
    "No screening": np.zeros(len(y_test), dtype=bool),
    "Built-in flag (isFlaggedFraud)": builtin_flag[~train_mask].to_numpy() == 1,
    "200,000 TRANSFER rule": rule.to_numpy() == 1,
    "Two-question tree": tree_flags == 1,
    f"XGBoost, 99%-precision cut-off ({operating_threshold:.2f})": xgb_scores >= operating_threshold,
    f"XGBoost, cheapest cut-off ({cutoffs[cheapest]:.2f})": xgb_scores >= cutoffs[cheapest],
}

# An empty dictionary, one outcome per screen
outcomes = {}
# Count the outcome of each screen
for name, screen_flags in screens.items():
    outcomes[name] = screen_outcome(screen_flags)
# Turn the outcomes into a table, one row per screen
impact = pd.DataFrame.from_dict(outcomes, orient="index")
# Type and size of the table
print(type(impact), impact.shape)
""")

code("""
# A copy of the table for display
shown = impact.copy()
# Show the value share as a percentage
shown["share of fraud value stopped"] = shown["share of fraud value stopped"].map("{:.2%}".format)
# Display the table
shown
""")

code("""
# Check your work
# The row name of XGBoost at the 99%-precision cut-off
model_screen = f"XGBoost, 99%-precision cut-off ({operating_threshold:.2f})"
# The model stops more value than the built-in flag
assert impact.loc[model_screen, "fraud value stopped"] > impact.loc["Built-in flag (isFlaggedFraud)", "fraud value stopped"], "the model should beat the built-in flag"
# No screen stops more than all the fraud
assert (impact["share of fraud value stopped"] <= 1).all(), "no screen can stop more than all the fraud"
# The built-in flag catches 10 test frauds
assert impact.loc["Built-in flag (isFlaggedFraud)", "frauds caught"] == 10, "the built-in flag should catch 10 test frauds"
""")

md("""
### Step 3.4: Chart value stopped against honest customers frozen

**Concept: a trade-off chart.** Two panels with the screens in the same order let a manager read the benefit and
the cost of each screen side by side.
""")

code("""
# A wide figure: value stopped on the left, customers frozen on the right
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

# Each screen's share of fraud value stopped, in percent
value_share_pct = impact["share of fraud value stopped"] * 100
# One bar per screen
value_share_pct.plot(kind="barh", ax=axes[0], color="#2e5c8a")
# Label the value chart's x-axis
axes[0].set_xlabel("Share of test-window fraud value stopped (%)")
# Title of the value chart
axes[0].set_title("Fraud value stopped")
# Write each share beside its bar
for bar_position, bar_value in enumerate(value_share_pct):
    axes[0].text(bar_value + 1, bar_position, f"{bar_value:.1f}%", va="center", fontsize=9)
# Room on the right for the share labels
axes[0].set_xlim(0, 115)

# Right: honest customers frozen by each screen
impact["honest customers frozen"].plot(kind="barh", ax=axes[1], color="#c0504d")
# Label the frozen chart's x-axis
axes[1].set_xlabel("Honest customers frozen (false alarms)")
# Title of the frozen chart
axes[1].set_title("The cost: honest customers frozen")
# Write each count beside its bar
for bar_position, bar_value in enumerate(impact["honest customers frozen"]):
    axes[1].text(bar_value + 2, bar_position, f"{bar_value:,}", va="center", fontsize=9)
# Drop the repeated screen names on the right panel
axes[1].set_yticklabels([])

# Keep the two impact panels apart
plt.tight_layout()
# Save them as figs/fraud_value_by_screen.png
plt.savefig("figs/fraud_value_by_screen.png", dpi=150, bbox_inches="tight")
# Show the impact chart
plt.show()
""")

code("""
# Check your work
# The impact chart was saved
assert os.path.isfile("figs/fraud_value_by_screen.png"), "figs/fraud_value_by_screen.png was not written"
""")

md("""
The built-in flag stops 1.11% of the fraud value. The 200,000 rule and the two-question tree each stop about half
(49.67% and 49.45%), at very different prices: the rule freezes 20,797 honest customers, the tree 79. XGBoost at
the 99%-precision cut-off stops 99.90% of the fraud value and freezes 27 honest customers; at the cheapest
cut-off it stops the same value and freezes 2. For a fraud team, this table is the model: the score matters only
through what it stops and what it costs.

### Step 3.5: Show the model's drivers in plain words

**Concept: drivers.** The inputs that move the score most, on average. A manager does not need SHAP; they need
to know what the model is looking at.
""")

code("""
# Importance from Step 2.27, smallest at the bottom of the chart
drivers = importance.sort_values()

# A blank chart for the drivers
fig, ax = plt.subplots(figsize=(9, 4.5))
# One horizontal bar per input
drivers.plot(kind="barh", ax=ax, color="#2e5c8a")
# Label the x-axis with the measure
ax.set_xlabel("Average push on the score (mean |SHAP|)")
# Title of the drivers chart
ax.set_title("What the XGBoost model looks at most")
# Keep the drivers chart inside its margins
plt.tight_layout()
# Show the drivers chart
plt.show()
# Type and length: one value per input
print(type(drivers), drivers.shape)
""")

code("""
# Check your work
# Nine inputs on the chart
assert len(drivers) == 9, f"expected 9 inputs, got {len(drivers)}"
# The top driver is the sender's new balance
assert drivers.index[-1] == "newBalanceOrig", f"the top driver changed to {drivers.index[-1]}"
""")

md("""
In words: the model looks first at whether the sender's account was emptied (`newBalanceOrig`), then at whether
the sender's side of the ledger balances (`errorBalanceOrig`), then at the recipient's starting balance. The hour
and the transaction type matter least.

### Step 3.6: Connect this notebook to the shipped model

The model in `src/` keeps the ledger idea but drops the raw balances (Step 2.27 showed why). Run its tested
feature code, `engineer_features` in `src/features.py`, on the two frauds from Step 1.5, then read the shipped
model's holdout results from the files the pipeline writes.

**Concept: one source of truth.** The feature code lives once, in `src/`, with unit tests. The notebook imports
it rather than copying it, so the two cannot drift apart.
""")

code("""
# Let Python find the modules in src/
sys.path.insert(0, "src")
# The tested feature code and the list of shipped inputs
from features import FEATURE_COLS, engineer_features

# The shipped features for rows 2 and 3
shipped_examples = engineer_features(raw_examples)
# Type and size of the shipped features
print(type(shipped_examples), shipped_examples.shape)
# Show the five shipped inputs for rows 2 and 3
shipped_examples[FEATURE_COLS]
""")

code("""
# Check your work
# Row 3's recipient gap is 21,363
assert shipped_examples.loc[3, "dest_balance_discrepancy"] == 21_363, "row 3's recipient gap should be 21,363"
# Row 2 drains the sender's whole balance
assert shipped_examples.loc[2, "orig_drain_ratio"] > 0.99, "row 2 should drain the sender's whole balance"
""")

md("""
The shipped `dest_balance_discrepancy` for row 3 is the same 21,363 as this notebook's `errorBalanceDest`, and
`orig_drain_ratio` is 1 for both rows: each fraud moved the sender's whole balance.

Now read the shipped model's results. `outputs/holdout_metrics.json` is written from the saved model on the same
132,136 test rows; `dashboard/data/business_impact.json` is written by `src/business_impact.py`.
""")

code("""
# json: read the pipeline's files and write this notebook's results
import json

# The shipped model's holdout metrics
with open("outputs/holdout_metrics.json", encoding="utf-8") as handle:
    shipped_holdout = json.load(handle)

# The shipped model's value stopped
with open("dashboard/data/business_impact.json", encoding="utf-8") as handle:
    shipped_impact = json.load(handle)["results"]["Model, threshold 0.8852"]

# Print the holdout size
print(f"holdout rows {shipped_holdout['holdout_rows']:,}, frauds {shipped_holdout['holdout_frauds']:,}")
# Print the caught, missed and false-alarm counts
print(f"caught {shipped_holdout['frauds_caught']:,}, missed {shipped_holdout['frauds_missed']}, false alarms {shipped_holdout['false_alarms']}")
# Print the PR-AUC and the value share
print(f"PR-AUC {shipped_holdout['pr_auc']:.4f}, share of fraud value stopped {shipped_impact['share_of_fraud_value_stopped']:.2%}")
""")

code("""
# Check your work
# The same 132,136 test rows
assert shipped_holdout["holdout_rows"] == len(X_test), "the shipped model should be scored on the same test rows"
# 2,742 caught and 3 false alarms
assert shipped_holdout["frauds_caught"] == 2_742 and shipped_holdout["false_alarms"] == 3, "shipped holdout counts changed"
""")

md("""
The shipped model catches 2,742 of the 2,754 test frauds, raises 3 false alarms, and stops 99.98% of the fraud
value, with five inputs and no raw balances. How it differs from this notebook:

| | This notebook | Shipped pipeline (`src/`) |
|---|---|---|
| Inputs | 9, including the raw balances and the hour | 5: the amount, the two balance errors, and two ratios (how much of the sender's balance moved, and the amount against the recipient's balance) |
| Raw balances | Kept, to show what they do | Removed: the model used "balance hits zero" as a shortcut |
| Settings | Library defaults, class weight 482 | Tuned on earlier windows only (35 trees, class weight 2.14) |
| Probabilities | Raw model scores | Calibrated (isotonic) on the training hours just before the test window, so scores read as the fraud frequencies seen there |
| Cut-off | Chosen on the test window, for teaching | Cheapest at 1,000 units per missed fraud and 10 per false alarm (assumed), chosen on calibration hours before the test window |
| Over time | Four test periods, cut-off fixed | Four expanding retraining windows, plus input-drift monitoring |

### What a fraud team should do with this

1. **Screen TRANSFER and CASH_OUT only, on the ledger errors.** On this simulated data a two-question rule on the ledger errors already
   stops half the fraud value for 79 false alarms; a boosted model on the same idea stops almost all of it.
2. **Set the cut-off from your own costs**, and restate precision at your real fraud rate before sizing the
   review queue (Step 2.15: 93.4% here, not 99%).
3. **Do not ship the raw balances.** They let the model flag every account that is emptied, including honest
   ones (Step 2.27).
4. **Re-check every period.** A model can collapse silently on later data (LightGBM, Step 2.19).

### Step 3.7: Write the results file

Every headline number above goes into `outputs/notebook_results.json`, with a short run record (the date, run
time, git commit, whether that code had uncommitted changes, and the Python and package versions), so the README
and `docs/METHODOLOGY.md` can be checked against it.

**Concept: provenance.** A number is only as good as the record of how it was made. The commit hash says exactly
which code produced it.

First, a small helper that asks git a question and returns nothing if git is not available.
""")

code("""
# platform: the Python version
import platform
# subprocess: run git from Python
import subprocess
# datetime: today's date in UTC
from datetime import datetime, timezone
# version: each installed library's version
from importlib.metadata import PackageNotFoundError, version


# A function that runs one git command and returns what it printed
def git(*args):
    try:
        # Run git and return what it printed
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        # git is not installed, or this folder is not a git repository
        return None


# Print the commit this run used
print("git commit:", git("rev-parse", "--short", "HEAD"))
""")

md("""
Next, the run record.
""")

code("""
# "git status --porcelain" prints one line per changed file and nothing if the code is clean. The notebook,
# figures and outputs are left out because running this notebook rewrites them.
changed_files = git("status", "--porcelain", "--", ".", ":(exclude)*.ipynb", ":(exclude)figs", ":(exclude)outputs")
# True when code changed since the commit; None when git is missing
if changed_files is None:
    uncommitted_code_changes = None
else:
    uncommitted_code_changes = len(changed_files) > 0

# The version of each library that produced the numbers
package_names = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "shap", "matplotlib", "seaborn"]
# An empty dictionary for the versions
package_versions = {}
# Look up each library's version
for package_name in package_names:
    try:
        package_versions[package_name] = version(package_name)
    except PackageNotFoundError:
        package_versions[package_name] = None

# Date, run time, commit, clean or not, Python and packages
run_record = {
    "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "runtime_minutes": round((time.time() - RUN_STARTED) / 60),
    "git_commit": git("rev-parse", "HEAD"),
    "uncommitted_code_changes": uncommitted_code_changes,
    "python": platform.python_version(),
    "packages": package_versions,
}
# Print the run record
print(json.dumps(run_record, indent=2))
""")

md("""
Then turn the model scores and SHAP importances into plain numbers that JSON can store.
""")

code("""
# An empty dictionary, one entry per model
model_metrics = {}
# Copy every metric as a plain float
for model_name in results:
    model_metrics[model_name] = {}
    for metric_name, value in results[model_name].items():
        model_metrics[model_name][metric_name] = float(value)

# An empty dictionary for the mean |SHAP| per input
shap_mean_abs = {}
# Copy each input's importance as a plain float
for feature_name, value in importance.items():
    shap_mean_abs[feature_name] = float(value)


# A function JSON uses for numpy numbers it cannot write itself
def to_json_value(value):
    # numpy numbers have .item() to turn them into plain Python numbers; anything else is written as text
    if hasattr(value, "item"):
        return value.item()
    return str(value)


# How many models and inputs will be written
print(len(model_metrics), "models,", len(shap_mean_abs), "inputs")
""")

md("""
Finally, assemble the file and write it.
""")

code("""
# Every result this notebook writes, grouped by part
notebook_results = {
    "provenance": run_record,
    "data": {
        "transactions": 6_362_620,
        "frauds": n_fraud,
        "fraud_rate": n_fraud / 6_362_620,
        "by_type": by_type.reset_index().to_dict(orient="records"),
        "transfer_and_cash_out": int(len(features)),
        "builtin_flag_flagged": int(flagged.sum()),
        "builtin_flag_recall": caught / n_fraud,
        "balance_patterns_pct": (balance_patterns * 100).round(2).to_dict(),
    },
    "split": {
        "split_step": SPLIT_STEP,
        "train_rows": int(len(X_train)), "train_fraud_rate": float(y_train.mean()), "train_frauds": int(y_train.sum()),
        "test_rows": int(len(X_test)), "test_fraud_rate": float(y_test.mean()), "test_frauds": int(y_test.sum()),
        "class_weight": float(class_weight),
        "model_inputs": MODEL_COLS,
    },
    "models": model_metrics,
    "cut_off": {
        "precision_floor": 0.99,
        "operating_threshold": operating_threshold,
        "precision": float(precision[best]), "recall": float(recall[best]),
        "frauds_missed": int(cm[1, 0]), "false_alarms": int(cm[0, 1]),
        "cost_ratio": COST_RATIO, "cheapest_threshold": float(cutoffs[cheapest]),
        "cheapest_missed": int(missed[cheapest]), "cheapest_false_alarms": int(false_alarms[cheapest]),
        "real_fraud_rate": float(real_fraud_rate),
        "false_alarms_per_million_at_real_rate": float(false_alarms_per_million),
        "precision_at_real_fraud_rate": float(precision_at_real_rate),
    },
    "periods": periods.reset_index().to_dict(orient="records"),
    "impact": {"test_fraud_value": float(total_fraud_value), "screens": impact.to_dict(orient="index")},
    "shap_mean_abs": shap_mean_abs,
}

# Write the results file
with open("outputs/notebook_results.json", "w", encoding="utf-8") as f:
    json.dump(notebook_results, f, indent=2, default=to_json_value)
# Confirm the file was written
print("Wrote outputs/notebook_results.json with", len(notebook_results), "sections")
""")

code("""
# Check your work
# Read the file back
with open("outputs/notebook_results.json", encoding="utf-8") as handle:
    written = json.load(handle)
# It holds the test-window fraud count
assert written["split"]["test_frauds"] == 2_754, "the results file should hold the test-window fraud count"
# It holds all seven models
assert len(written["models"]) == 7, f"expected 7 models in the results file, got {len(written['models'])}"
""")

md("""
---
## What you built

- A clean, checked table of 6,362,620 transactions, narrowed to the 2,770,409 TRANSFER and CASH_OUT rows where all
  8,213 frauds are (Part 1).
- Two features from one accounting rule, a leakage guard that runs, and a time split that tests on the future.
- A baseline (PR-AUC 0.0283), a readable two-question tree (0.4892), and XGBoost (0.9989), with three more model
  families and a resampling check scored the same way (Part 2).
- Two cut-off policies, and precision restated at the real fraud rate (93.4%).
- A period-by-period check and SHAP explanations of single decisions.
- A business-impact table: XGBoost stops 99.90% of the test-window fraud value and freezes 2 honest customers at
  the cheapest cut-off, against 1.11% for the built-in flag (Part 3).
- A results file with a run record, so every number above can be traced.

## Limitations

- **PaySim is a simulator, and no real operator's data was used.** Once the ledger errors exist, its fraud labels follow from them almost exactly,
  which is why a random forest reaches PR-AUC 1.0000. The scores show that the method works; they say nothing
  about the accuracy to expect on a real network.
- **One fraud signature.** 97.82% of simulated frauds drain the sender's whole balance. Partial-skim fraud, where
  only part of a balance is taken, is barely represented and would not be detected.
- **The raw balances are a shortcut.** This notebook keeps them to show what they do; a customer honestly closing
  their own account would be flagged. The shipped model removes them and only partly closes the gap.
- **The test window is fraud-heavy.** 2.08% of its transactions are fraud against about 0.30% across the month,
  because honest volume collapses late in the simulation. Every precision figure depends on that mix.
- **The cut-offs here are chosen on the test window.** That is fine for teaching and not for deployment; the
  shipped pipeline chooses its cut-off on calibration data that precedes the test window.
- **Money figures are simulated units**, and the costs behind the cut-offs (10 units per missed fraud and 1 per
  false alarm here; 1,000 and 10 in the shipped pipeline) are assumptions, not an operator's figures.
- **One later window, no retraining.** This notebook's model is tested once on the hours after 490, split into
  four periods. The shipped pipeline's walk-forward check (`src/validate.py`) retrains on four expanding windows.
- **No account history.** Almost every account appears once, so the features a real fraud team relies on
  (velocity, repeat recipients, device and location) cannot be built from this data.

## Next steps

1. Remove `oldBalanceOrig` and `newBalanceOrig` from `MODEL_COLS`, re-run from Step 1.27, and compare XGBoost's
   PR-AUC and the SHAP chart with the version above.
2. Change `SPLIT_STEP` to 600 in Step 1.2. How do the training fraud rate and the test results change?
3. In Step 2.15, set `real_fraud_rate` to 0.005 and to 0.0005. How does precision move, and what does that mean
   for a team that can review 50 alerts a day?
4. Repeat the analysis on real transaction data with account history, where a mule-account feature becomes
   possible.

## Glossary

- **Precision / recall:** of the alarms raised, the share that are fraud / of the frauds, the share caught.
- **PR-AUC (average precision):** one number summarising precision and recall over every cut-off; random
  guessing scores the fraud rate.
- **ROC-AUC:** the chance a random fraud scores above a random honest transaction; looks high even for weak fraud
  models because honest transactions are so many.
- **Base rate:** the share of fraud in a group of transactions.
- **Class weight:** how much more a mistake on the rare class counts during training.
- **Cut-off (threshold):** the score above which a transaction is flagged.
- **Leakage:** a model seeing information it would not have at prediction time.
- **SMOTE:** a resampling method that creates synthetic examples of the rare class (Chawla et al., 2002).
- **SHAP value:** one input's contribution to one prediction, against an average prediction (Lundberg and Lee,
  2017).
- **Time split / walk-forward validation:** testing only on transactions later than every training row; walk-forward
  repeats that over several successive later windows, retraining each time.
- **Calibration:** mapping raw model scores to probabilities that match how often fraud actually occurred at each
  score.
- **PSI (Population Stability Index):** a measure of how much a feature's distribution in a new window differs from
  the training window; below 0.10 is usually read as stable, 0.25 or more as a shift to investigate.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
with open("Fraud_Detection_System.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote Fraud_Detection_System.ipynb with {len(cells)} cells")
