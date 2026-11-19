.PHONY: help setup install download-datasets verify-datasets prepare-data train evaluate test lint format clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

help:
	@echo "Kidney-AI Management Commands:"
	@echo "  make setup              - Set up Python virtual environment and folders"
	@echo "  make install            - Install production & dev dependencies"
	@echo "  make download-datasets  - Run dataset downloader script (dry-run/fetch)"
	@echo "  make verify-datasets    - Verify dataset integrity, clinical bounds, and checksums"
	@echo "  make prepare-data       - Execute data cleaning, feature engineering & splits"
	@echo "  make train              - Train all classical & deep learning kidney models"
	@echo "  make evaluate           - Run full evaluation benchmark and metrics"
	@echo "  make test               - Run unit and integration tests"
	@echo "  make lint               - Run code quality linter (ruff / flake8)"
	@echo "  make format             - Auto-format code using black and ruff"
	@echo "  make clean              - Clean build artifacts, pycache and temp logs"

setup:
	$(PYTHON) scripts/setup_environment.py

install:
	$(PIP) install -r requirements-dev.txt

download-datasets:
	$(PYTHON) scripts/download_datasets.py --mode sample

verify-datasets:
	$(PYTHON) scripts/verify_datasets.py

prepare-data:
	$(PYTHON) scripts/prepare_data.py

train:
	$(PYTHON) scripts/train_all.py --config configs/training.yaml

evaluate:
	$(PYTHON) scripts/evaluate_all.py

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

format:
	$(PYTHON) -m black src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
