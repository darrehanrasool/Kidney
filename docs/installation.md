# Installation & Environment Configuration

## System Prerequisites
- Python 3.10, 3.11, or 3.12
- macOS (Apple Silicon MPS supported), Linux (Ubuntu 20.04+, CUDA 12+), or Windows 11 (WSL2 recommended)

## Quick Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   make install
   ```

3. Verify hardware accelerator:
   ```bash
   python scripts/verify_hardware.py
   ```
