#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = fmnist_classification
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 src
	isort --check --diff src
	black --check src

## Format source code with black
.PHONY: format
format:
	isort src
	black src





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

BATCH_SIZE = 64
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 10
MODEL_PATH = models

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

.PHONY: train

train:
	$(PYTHON_INTERPRETER) -m src.modeling.train \
		--batch-size $(BATCH_SIZE) \
		--stage1-epochs $(STAGE1_EPOCHS) \
		--stage2-epochs $(STAGE2_EPOCHS)


help:
	@echo "Available commands:"
	@echo ""
	@echo "Training:"
	@echo "  make train                    - Train the Fashion-MNIST model with default parameters"
	@echo "  make train BATCH_SIZE=32      - Train with custom batch size"
	@echo "  make train STAGE1_EPOCHS=5    - Train with custom stage 1 epochs"
	@echo "  make train STAGE2_EPOCHS=15   - Train with custom stage 2 epochs"
	@echo ""
	@echo "Parameters:"
	@echo "  BATCH_SIZE     = $(BATCH_SIZE)     (Training batch size)"
	@echo "  STAGE1_EPOCHS  = $(STAGE1_EPOCHS)  (Epochs for stage 1 - classifier only)"
	@echo "  STAGE2_EPOCHS  = $(STAGE2_EPOCHS)  (Epochs for stage 2 - fine-tuning)"
	@echo ""
	@echo "Examples:"
	@echo "  make train BATCH_SIZE=32 STAGE1_EPOCHS=5"
	@echo "  make train STAGE2_EPOCHS=15"
