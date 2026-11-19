# Contributing to Kidney-AI

Thank you for your interest in contributing to the **Kidney-AI** research platform! We welcome contributions from data scientists, clinical researchers, nephrologists, and machine learning engineers.

## Development Principles & Standards

1. **Rigorous Reproducibility**: All experiments must be reproducible with fixed seeds, logged configs, and deterministic settings.
2. **Clinical Safety First**: Models must output calibrated probabilities, confidence bounds, and explainability indicators. Never commit patient identifiers or raw Protected Health Information (PHI).
3. **Clean Code & Type Safety**: Follow PEP 8 guidelines enforced via `ruff` and `black`. Use type hints for all public functions and classes.
4. **Comprehensive Testing**: Any new pipeline feature, data preprocessor, or model architecture must include matching unit and integration tests.

## Getting Started

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/darrehanrasool/Kidney.git
   cd Kidney
   ```

2. Set up the environment:
   ```bash
   make setup
   make install
   ```

3. Run verification tests:
   ```bash
   make test
   make lint
   ```

## Pull Request Checklist

Before submitting a PR:
- [ ] Code is formatted with `black` and passes `ruff` checks (`make format && make lint`).
- [ ] Unit and integration tests pass cleanly (`make test`).
- [ ] Configuration files in `configs/` are updated if new parameters are introduced.
- [ ] Documentation and model cards in `docs/` reflect any methodological changes.
- [ ] No raw dataset files or large model binary checkpoints are added to git.
