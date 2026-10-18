# Troubleshooting & Common Issues Guide

## 1. Low-Memory Environments & RAM Constraints
- **Symptom**: Out-of-memory errors, system thrashing, or process termination during training on machines with limited RAM (e.g. 8GB or 16GB laptops).
- **Resolution**:
  1. Set `num_workers: 0` in `configs/training.yaml` to eliminate multiprocessing subprocess memory overhead.
  2. Maintain `batch_size: 32` or reduce to `16` in `configs/training.yaml` for neural networks.
  3. Set `preferred_device: "cpu"` in `configs/hardware.yaml` if Apple Silicon unified memory pressure is high.
  4. Ensure `gc.collect()` and `torch.mps.empty_cache()` / `torch.cuda.empty_cache()` are called via `MemoryMonitor`.

## 2. Clinical Data Formatting & Missing Values
- **Symptom**: `ValueError: could not convert string to float: '?'` or `ckd\t`.
- **Cause**: Real-world medical datasets like the UCI Chronic Kidney Disease benchmark contain whitespace anomalies, trailing tabs (`ckd\t` instead of `ckd`, `\tno` instead of `no`), and question marks `?` representing unrecorded lab tests.
- **Resolution**:
  1. `src/kidneycare/data/data_validator.py` and `preprocessing.py` automatically strip whitespace, clean trailing tabs, and replace `?` with `np.nan`.
  2. Imputation is managed transparently via `MissingValueImputer` using configurable median, mean, mode, or KNN strategies fit strictly on training splits.

## 3. Apple Silicon (MPS) Backend Compatibility
- **Symptom**: `UserWarning: The operator 'aten::...' is not currently implemented for the MPS device`.
- **Resolution**:
  1. Run with environment variable: `export PYTORCH_ENABLE_MPS_FALLBACK=1`.
  2. Alternatively, specify `--device cpu` in script arguments to force clean CPU execution.

## 4. Checkpoint Resumption & Interrupted Training
- **Symptom**: Training script was interrupted due to power loss, timeout, or manual abort.
- **Resolution**:
  Resume training from the latest valid checkpoint with:
  ```bash
  python3 scripts/resume_training.py --checkpoint-dir checkpoints/latest
  ```
  The `RecoveryManager` verifies checkpoint integrity, restores optimizer/scheduler states, and resumes at the subsequent epoch without repeating completed epochs.

## 5. Offline Operation Without Internet Access
- **Symptom**: Network error / DNS resolution failure when running `scripts/download_datasets.py` in an offline or sandboxed environment.
- **Resolution**:
  Use the synthetic benchmark fixture, which requires zero internet access:
  ```bash
  python3 scripts/download_datasets.py --mode synthetic
  ```
  This creates a high-fidelity 400-record clinical dataset matching the exact schema and correlations of the UCI CKD benchmark.
