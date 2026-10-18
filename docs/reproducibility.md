# Reproducibility Guide & Benchmark Auditability

## 1. Deterministic Seeding Protocol
To guarantee exact numerical replication across heterogeneous hardware architectures (CPU, Apple Silicon MPS, NVIDIA CUDA), KidneyCare-AI enforces strict seed synchronization across all random number generators via `src/kidneycare/core/reproducibility.py`:

```python
from kidneycare.core.reproducibility import set_seed, get_environment_telemetry

# Set fixed seed and configure deterministic execution
set_seed(42, deterministic_torch=True)

# Collect environment telemetry for audit logs
telemetry = get_environment_telemetry()
```

This synchronization performs:
1. `random.seed(seed)`
2. `np.random.seed(seed)`
3. `torch.manual_seed(seed)`
4. `torch.cuda.manual_seed_all(seed)` (when CUDA is available)
5. `torch.backends.cudnn.deterministic = True`
6. `torch.backends.cudnn.benchmark = False`
7. Sets environment variable `PYTHONHASHSEED = str(seed)`

## 2. Leakage-Free Preprocessing Architecture
Data leakage is one of the most pervasive failure modes in clinical machine learning. KidneyCare-AI strictly implements the scikit-learn Pipeline architecture to prevent any information leakage:
- **Imputation**: Median/KNN values are computed exclusively from the training split.
- **Scaling**: RobustScaler / StandardScaler parameters (medians, IQR, means, variances) are fit strictly on training partitions and applied to validation/test partitions without refitting.
- **Categorical Encoding**: Target or one-hot encodings are learned solely on the training partition. Unseen categories in validation/test are mapped to unknown/missing category buckets.
- **Threshold Optimization**: Decision thresholds (maximizing Youden's $J$-statistic or $F_1$-score) are selected strictly on the validation set and evaluated once on the hold-out test set.

## 3. Dataset Integrity Manifests & Provenance
- Every raw clinical dataset archive is verified against SHA-256 digests in `data/manifests/dataset_manifest.json`.
- Streaming checksum calculations prevent loading massive raw files into memory during integrity verification.
- Fitted preprocessor state is serialized to `models/preprocessors/ckd_preprocessor.joblib` to guarantee identical inference-time data transformations.

## 4. Hardware Independence & Telemetry
Every training run logs an experiment metadata JSON containing:
- Operating system, kernel version, and processor architecture.
- Python version, PyTorch version, Scikit-learn version.
- Active execution backend (`mps`, `cuda`, or `cpu`).
- Git commit hash and repository status.
