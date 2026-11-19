# Kidney-AI: Researcher Operational Guide & Execution Playbook

This document provides researchers with end-to-end instructions for data ingestion, training, evaluation, XAI generation, and deployment of the Kidney-AI platform.

## 1. Environment Initialization
Run the initialization script to ensure all directories, logs, checkpoints, and split folders exist:
```bash
make setup
make install
```

## 2. Hardware Verification
Verify that your compute accelerator (MPS on Apple Silicon, CUDA on Linux/Windows, or multi-threaded CPU) is detected:
```bash
python src/kidney_ai/main.py verify-hardware
```

## 3. Data Ingestion & Physiological Audit
Download or generate the verified benchmark cohort:
```bash
python src/kidney_ai/main.py download --mode sample
python src/kidney_ai/main.py verify-datasets
```

## 4. Feature Engineering & Splitting
Execute eGFR calculation, BUN/Creatinine ratios, scaling, and stratified splitting:
```bash
python src/kidney_ai/main.py prepare
```

## 5. Model Zoo Training
Train classical estimators and deep tabular residual neural networks:
```bash
python src/kidney_ai/main.py train
```

## 6. Benchmarking & Figure Generation
Evaluate performance across all metrics on the test partition and plot figures:
```bash
python src/kidney_ai/main.py evaluate
python src/kidney_ai/main.py visualize
```

## 7. Interactive Clinical Prediction
Test unified pipeline inference on sample patients:
```bash
python src/kidney_ai/main.py pipeline
```
