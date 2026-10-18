.PHONY: help setup install download-datasets verify-datasets prepare-data train evaluate visualize test clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

help:
	@echo "KidneyCare-AI Management Commands:"
	@echo "  make setup              - Set up project directory structure and environment"
	@echo "  make install            - Install production & development dependencies"
	@echo "  make download-datasets  - Run dataset downloader script"
	@echo "  make verify-datasets    - Verify dataset integrity, readability and checksums"
	@echo "  make prepare-data       - Execute data sanitization, clinical feature engineering & splits"
	@echo "  make train              - Train all classical & deep learning CKD detection models"
	@echo "  make evaluate           - Run full evaluation benchmark, calibration, and metrics"
	@echo "  make visualize          - Generate all 10+ clinical evaluation and data visualizations"
	@echo "  make test               - Run unit and integration tests"
	@echo "  make clean              - Clean build artifacts, pycache and temp caches"

setup:
	$(PYTHON) scripts/setup_environment.py

install:
	$(PIP) install -e .

download-datasets:
	$(PYTHON) scripts/download_datasets.py --source all

verify-datasets:
	$(PYTHON) scripts/verify_datasets.py

prepare-data:
	$(PYTHON) scripts/prepare_data.py

train:
	$(PYTHON) scripts/train_all.py --config configs/training.yaml

evaluate:
	$(PYTHON) scripts/evaluate_all.py

visualize:
	$(PYTHON) scripts/generate_visualizations.py

test:
	$(PYTHON) -m pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
