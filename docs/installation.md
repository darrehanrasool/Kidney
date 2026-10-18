# Installation and Environment Setup Guide

## 1. System Requirements

- **Python**: Version 3.10, 3.11, 3.12, 3.13, or 3.14.
- **Operating Systems**: macOS (Apple Silicon M1–M4 or Intel x86_64), Linux (Ubuntu 20.04+, RHEL 8+, Debian), or Windows 10/11.
- **Hardware Acceleration**:
  - **macOS**: Apple Silicon Metal Performance Shaders (MPS) auto-detected.
  - **Linux / Windows**: NVIDIA CUDA 11.8+ / 12.0+ auto-detected.
  - **CPU-Only**: Fully supported with low-memory footprint ($<500\text{ MB}$ RAM).

---

## 2. Installation Walkthrough

### Step 1: Navigate to the Project Root
```bash
cd Kidney
```

### Step 2: Create and Activate a Clean Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install KidneyCare-AI Dependencies
```bash
pip install --upgrade pip
pip install -e .
```
For developer testing and coverage tools:
```bash
pip install -r requirements-dev.txt
```

### Step 4: Verify Hardware Detection & Runtime Backends
```bash
python3 scripts/verify_hardware.py
```
Outputs CPU cores, available system RAM, and detected accelerators (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback).

---

## 3. End-to-End Workflow Commands

### Step 1: Ingest or Generate Clinical Dataset
To download the canonical UCI Chronic Kidney Disease dataset:
```bash
python3 scripts/download_datasets.py --mode download
```
For offline zero-network environments:
```bash
python3 scripts/download_datasets.py --mode synthetic
```

### Step 2: Validate Data Integrity & Clinical Ranges
```bash
python3 scripts/verify_datasets.py
```

### Step 3: Sanitize, Feature Engineer, and Create Splits
```bash
python3 scripts/prepare_data.py
```

### Step 4: Train All Clinical Models
```bash
python3 scripts/train_all.py
```

### Step 5: Benchmark Evaluation on Holdout Test Partition
```bash
python3 scripts/evaluate_all.py
```

### Step 6: Generate All 10+ Publication Visualizations
```bash
python3 scripts/generate_visualizations.py
```

### Step 7: Run Unit & Integration Test Suite
```bash
pytest tests/ -v
```
