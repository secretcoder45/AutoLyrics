# =============================================================
#  AutoLyrics — Makefile
# =============================================================

PYTHON      ?= python
PIP         ?= pip
PROJECT     := autolyrics
SRC_DIR     := src/$(PROJECT)
TEST_DIR    := tests
SCRIPT_DIR  := scripts
DATA_DIR    ?= data
RUN_DIR     ?= runs

.PHONY: help install install-dev clean clean-pyc clean-build clean-test \
        lint format type-check test test-unit test-integration test-cov \
        download preprocess train-baseline train-lora train-lora-encdec \
        train-qlora evaluate-baseline evaluate-lora benchmark report \
        demo api docker docker-build docker-up docker-down precommit

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ----- Install ------------------------------------------------
install:  ## Install runtime dependencies and package.
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev:  ## Install dev dependencies (lint, test, type-check).
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install || true

# ----- Quality ------------------------------------------------
lint:  ## Run ruff lint.
	ruff check $(SRC_DIR) $(TEST_DIR) $(SCRIPT_DIR)

format:  ## Auto-format with ruff.
	ruff format $(SRC_DIR) $(TEST_DIR) $(SCRIPT_DIR)
	ruff check --fix $(SRC_DIR) $(TEST_DIR) $(SCRIPT_DIR)

type-check:  ## Run mypy.
	mypy $(SRC_DIR)

precommit:  ## Run all pre-commit hooks on all files.
	pre-commit run --all-files

# ----- Test ---------------------------------------------------
test:  ## Run full test suite.
	pytest $(TEST_DIR)

test-unit:  ## Run unit tests only (fast, no GPU).
	pytest $(TEST_DIR)/unit -m "not slow and not gpu"

test-integration:  ## Run integration tests.
	pytest $(TEST_DIR)/integration

test-cov:  ## Run tests with coverage report.
	pytest --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html $(TEST_DIR)

# ----- Data ---------------------------------------------------
download:  ## Download all supported datasets.
	$(PYTHON) $(SCRIPT_DIR)/download_datasets.py --dataset all --output $(DATA_DIR)/raw

preprocess:  ## Preprocess datasets (resample + feature cache).
	$(PYTHON) $(SCRIPT_DIR)/preprocess_data.py --config configuration/nus48e.yaml

# ----- Training -----------------------------------------------
train-lora:  ## LoRA decoder-only fine-tune on Whisper-small / NUS-48E.
	$(PYTHON) $(SCRIPT_DIR)/train.py \
	    --model-config configuration/whisper_small.yaml \
	    --data-config configuration/nus48e.yaml \
	    --training-config configuration/lora_decoder.yaml \
	    --output-dir $(RUN_DIR)/lora_decoder_nus48e

train-lora-encdec:  ## LoRA encoder + decoder fine-tune.
	$(PYTHON) $(SCRIPT_DIR)/train.py \
	    --model-config configuration/whisper_small.yaml \
	    --data-config configuration/nus48e.yaml \
	    --training-config configuration/lora_encode_decoder.yaml \
	    --output-dir $(RUN_DIR)/lora_encdec_nus48e

train-qlora:  ## QLoRA (4-bit) on Whisper-medium.
	$(PYTHON) $(SCRIPT_DIR)/train.py \
	    --model-config configuration/whisper_medium.yaml \
	    --data-config configuration/nus48e.yaml \
	    --training-config configuration/qlora_decoder.yaml \
	    --output-dir $(RUN_DIR)/qlora_medium_nus48e

# ----- Evaluation ---------------------------------------------
evaluate-baseline:  ## Zero-shot Whisper-small baseline on NUS-48E.
	$(PYTHON) $(SCRIPT_DIR)/evaluate.py \
	    --model-config configuration/whisper_small.yaml \
	    --data-config configuration/nus48e.yaml \
	    --output reports/baseline_nus48e.json

evaluate-lora:  ## Evaluate latest LoRA checkpoint.
	$(PYTHON) $(SCRIPT_DIR)/evaluate.py \
	    --checkpoint $(RUN_DIR)/lora_decoder_nus48e/best \
	    --data-config configuration/nus48e.yaml \
	    --output reports/lora_decoder_nus48e.json

benchmark:  ## Benchmark latency and VRAM.
	$(PYTHON) $(SCRIPT_DIR)/benchmark.py \
	    --checkpoint $(RUN_DIR)/lora_decoder_nus48e/best \
	    --batch-sizes 1 4 8 \
	    --output reports/benchmark.json

report:  ## Build the comparative PDF performance report.
	$(PYTHON) $(SCRIPT_DIR)/generate_report.py \
	    --baseline reports/baseline_nus48e.json \
	    --lora-decoder reports/lora_decoder_nus48e.json \
	    --benchmark reports/benchmark.json \
	    --output reports/performance_report.pdf

# ----- Serving ------------------------------------------------
demo:  ## Launch the Gradio demo.
	$(PYTHON) -m autolyrics.ui.gradio_app

api:  ## Launch the FastAPI service.
	uvicorn autolyrics.api.main:app --host 0.0.0.0 --port 8000 --reload

# ----- Docker -------------------------------------------------
docker-build:  ## Build the Docker image.
	docker build -t autolyrics:latest .

docker-up:  ## Bring up docker-compose stack.
	docker compose up --build

docker-down:  ## Tear down docker-compose stack.
	docker compose down -v

# ----- Cleanup ------------------------------------------------
clean: clean-pyc clean-build clean-test  ## Remove all build / test / cache artifacts.

clean-pyc:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-build:
	rm -rf build/ dist/ .eggs/

clean-test:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/