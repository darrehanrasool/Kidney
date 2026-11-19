# Experimental Reproducibility Protocol

## Seed & State Determinism
To guarantee identical metric outputs across repeated experimental runs:
- Global random seed is pinned to `42` across Python `random`, `numpy`, and `torch`.
- PyTorch execution flags:
  ```python
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

## Hardware Consistency
Environment checks automatically verify compute precision (`float32` vs. `mixed_float16`) and record device identifiers in experiment logs under `logs/hardware/`.
