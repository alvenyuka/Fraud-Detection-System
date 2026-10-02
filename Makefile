# Fraud Detection System: task runner
#
# Usage:
#   make install          Install Python dependencies (exact pins)
#   make install-lock     Install the full locked dependency tree
#   make test             Run the unit tests (no dataset needed)
#   make tune             Search hyperparameters (writes model/best_params.json)
#   make validate         Walk-forward validation (writes dashboard/data artifacts)
#   make monitor          PSI drift monitoring (writes dashboard/data/psi_timeline.csv)
#   make explain          Feature importance / probability spread / threshold curve
#   make scenarios        Regenerate MODEL_CARD's scenario tables (no dataset needed)
#   make figures          Redraw the README chart from dashboard/data (no dataset needed)
#   make train            Train model on PaySim data
#   make predict          Score a single transaction (interactive)
#   make predict-csv      Score transactions.csv -> scored.csv
#   make dashboard        Launch the live Streamlit dashboard
#   make notebook         Launch Jupyter Lab
#   make clean            Remove __pycache__ and build artefacts

.PHONY: install install-lock test tune validate monitor explain scenarios figures train predict predict-csv dashboard notebook clean help

DATA     ?= PS_20174392719_1491204439457_log.csv
MODEL    ?= model/xgb_fraud_model.pkl
INPUT    ?= transactions.csv
OUTPUT   ?= scored.csv
PYTHON   := python

# -- Install -----------------------------------------------------------------

install:
	$(PYTHON) -m pip install -r requirements.txt

install-lock:
	$(PYTHON) -m pip install -r requirements-lock.txt

# -- Test --------------------------------------------------------------------

# Runs on small synthetic frames only, so it needs neither the 470MB PaySim CSV
# nor a trained model. This is the same command CI runs.
test:
	$(PYTHON) -m pytest

# -- Tune / Validate / Monitor -------------------------------------------------

tune: $(DATA)
	$(PYTHON) src/tune.py --data $(DATA)

validate: $(DATA)
	$(PYTHON) src/validate.py --data $(DATA)

monitor: $(DATA)
	$(PYTHON) src/monitoring.py --data $(DATA)

explain: $(DATA)
	$(PYTHON) src/explain.py --data $(DATA)

# Needs only the committed model artifact, not the dataset.
scenarios:
	$(PYTHON) src/scenarios.py

# Reads only dashboard/data, so it needs neither the dataset nor a training run.
figures:
	$(PYTHON) src/make_figures.py

# -- Train -------------------------------------------------------------------

train: $(DATA)
	@echo "Training on $(DATA) -> $(MODEL)"
	$(PYTHON) src/train.py --data $(DATA) --model-out $(MODEL)

$(DATA):
	@echo "Error: dataset not found at '$(DATA)'"
	@echo "Download from https://www.kaggle.com/datasets/ealaxi/paysim1"
	@echo "Then run: make train DATA=<path-to-csv>"
	@exit 1

# -- Predict -----------------------------------------------------------------

predict: $(MODEL)
	$(PYTHON) src/predict.py --model $(MODEL)

predict-csv: $(MODEL) $(INPUT)
	$(PYTHON) src/predict.py --model $(MODEL) --input $(INPUT) --output $(OUTPUT)
	@echo "Scored transactions saved to $(OUTPUT)"

$(MODEL):
	@echo "Error: model not found at '$(MODEL)'. Run 'make train' first."
	@exit 1

# -- Dashboard -----------------------------------------------------------------

dashboard:
	streamlit run dashboard/app.py

# -- Notebook ----------------------------------------------------------------

# Needs jupyterlab, which requirements.txt does not pin.
notebook:
	jupyter lab Fraud_Detection_System.ipynb

# -- Clean -------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# -- Help --------------------------------------------------------------------

help:
	@echo ""
	@echo "  make install          Install dependencies from requirements.txt"
	@echo "  make install-lock     Install the full locked dependency tree"
	@echo "  make test             Run unit tests (no dataset or model needed)"
	@echo "  make tune             Search hyperparameters -> model/best_params.json"
	@echo "  make validate         Walk-forward validation -> dashboard/data artifacts"
	@echo "  make monitor          PSI drift monitoring -> dashboard/data/psi_timeline.csv"
	@echo "  make explain          Feature importance / probability spread / threshold curve"
	@echo "  make figures          Redraw the README chart from dashboard/data"
	@echo "  make train            Train on PaySim CSV  (set DATA= to override path)"
	@echo "  make predict          Score one transaction interactively"
	@echo "  make predict-csv      Score INPUT csv -> OUTPUT csv"
	@echo "  make dashboard        Launch the live Streamlit dashboard"
	@echo "  make notebook         Open Jupyter Lab"
	@echo "  make clean            Remove __pycache__ files"
	@echo ""
	@echo "  Override variables:  make train DATA=data/paysim.csv MODEL=model/v2.pkl"
	@echo ""
