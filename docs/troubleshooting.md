# Troubleshooting & FAQ

### 1. PyTorch Apple Silicon MPS Issues
If running on macOS and encounters unsupported MPS operations:
- Set `device: cpu` in `configs/hardware.yaml` or pass `--device cpu` in the CLI.

### 2. Missing C Extensions or Build Tools
If installing `xgboost` or `scikit-learn` fails:
- Ensure Xcode command-line tools (macOS: `xcode-select --install`) or `build-essential` (Linux) are installed.

### 3. Missing Dataset Files
Run:
```bash
python scripts/download_datasets.py --mode sample
```
to automatically generate or download sample verified benchmark cohorts.
