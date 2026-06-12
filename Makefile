# Fraud Detection System — task runner
#
# Usage:
#   make install          Install Python dependencies
#   make train            Train model on PaySim data
#   make predict          Score a single transaction (interactive)
#   make predict-csv      Score transactions.csv -> scored.csv
#   make notebook         Launch Jupyter Lab
#   make clean            Remove __pycache__ and build artefacts

.PHONY: install train predict predict-csv notebook clean help

DATA     ?= PS_20174392719_1491204439457_log.csv
MODEL    ?= model/xgb_fraud_model.pkl
INPUT    ?= transactions.csv
OUTPUT   ?= scored.csv
PYTHON   := python

# -- Install -----------------------------------------------------------------

install:
	$(PYTHON) -m pip install -r requirements.txt

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

# -- Notebook ----------------------------------------------------------------

notebook:
	jupyter lab "Fraud Detection System.ipynb"

# -- Clean -------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# -- Help --------------------------------------------------------------------

help:
	@echo ""
	@echo "  make install          Install dependencies from requirements.txt"
	@echo "  make train            Train on PaySim CSV  (set DATA= to override path)"
	@echo "  make predict          Score one transaction interactively"
	@echo "  make predict-csv      Score INPUT csv -> OUTPUT csv"
	@echo "  make notebook         Open Jupyter Lab"
	@echo "  make clean            Remove __pycache__ files"
	@echo ""
	@echo "  Override variables:  make train DATA=data/paysim.csv MODEL=model/v2.pkl"
	@echo ""
