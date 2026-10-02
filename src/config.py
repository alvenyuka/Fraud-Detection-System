"""
config.py: the time windows every script in src/ agrees on.

These live in their own dependency-free module so the tests can import them
without optuna, xgboost or the dataset, and so the guard that keeps tuning rows
away from reporting rows (tests/test_split_and_monitoring.py) checks the same
numbers the scripts use.

PaySim's steps run from 1 to 743, one step per simulated hour.
"""

# Train on steps 1 to SPLIT_STEP, report on the holdout after it (steps 491-743).
SPLIT_STEP = 490

# Three "train up to A, score on A+1 to B" windows used only by src/tune.py. All
# of them end at or before step 350, below every reporting window.
TUNING_SPLITS = [
    (200, 250),
    (250, 300),
    (300, 350),
]

# Expanding-window folds used by src/validate.py: train on steps 1 to A, test on
# A+1 to B. Together they test steps 351 to 743.
FOLDS = [
    (350, 450),
    (450, 550),
    (550, 650),
    (650, 743),
]

# Share of a training period's steps, taken from its end, that is held out:
# for early stopping in src/tune.py, and for isotonic calibration and threshold
# selection in src/train.py and src/validate.py.
HOLDOUT_STEP_SHARE = 0.2
