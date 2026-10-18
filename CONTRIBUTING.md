# Contributing to KidneyCare-AI

Thank you for your interest in contributing to **KidneyCare-AI**! We are committed to developing open, reproducible, and clinically validated machine learning systems for kidney disease risk assessment, early detection, and progression monitoring.

---

## 1. Ethical Guidelines & Data Privacy (Strict Non-PHI Policy)

KidneyCare-AI is a clinical AI research framework. **Under no circumstances should Protected Health Information (PHI) or personally identifiable clinical data be committed to this repository.**

- **De-identification**: All data committed to fixtures, tests, or examples must be completely synthetic or sourced from approved, publicly de-identified datasets (e.g., UCI Machine Learning Repository, PhysioNet, MIMIC under DUA terms).
- **HIPAA & GDPR Compliance**: Respect all applicable data privacy and patient protection standards.
- **Reporting Exposure**: If you suspect any patient data or sensitive credentials have been committed, notify the security maintainers immediately at `security@kidneycare-ai.org`.

---

## 2. Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to ensure a collaborative, respectful, and safe community.

---

## 3. Development Workflow

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git 2.30+
- Virtual environment tool (`venv`, `conda`, or `uv`)

### Local Setup
1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/KidneyCare-AI.git
   cd KidneyCare-AI
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```
4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

---

## 4. Branching Strategy & Git Commit Conventions

- Create descriptive feature branches from `main`:
  - `feat/feature-name` for new models, features, or pipelines
  - `fix/issue-description` for bug fixes
  - `docs/topic` for documentation updates
  - `clinical/biomarker-update` for physiological rule updates
- Commit messages should follow the Conventional Commits format:
  ```text
  feat(models): add tabular MLP architecture with cosine annealing
  fix(preprocessing): correct eGFR lower bound clipping
  docs(architecture): update pipeline data flow diagram
  test(metrics): add unit test for expected calibration error (ECE)
  ```

---

## 5. Coding Standards

- **Formatting**: We use `black` with a 100-character line length limit.
- **Imports**: Sorted using `isort` with the `black` profile.
- **Linting**: Code must pass `flake8` checks without syntax or undefined variable warnings.
- **Type Annotations**: Provide Python type hints for all public functions, classes, and methods. Check types with `mypy`.
- **Documentation**: Use Google-style docstrings with explicit arguments, returns, and exception specifications.

---

## 6. Testing & Clinical Validation Requirements

- **Unit Tests**: All new algorithms, metrics, and preprocessing transforms must include corresponding tests under `tests/unit/`.
- **Integration Tests**: Changes affecting multi-step pipelines must pass `tests/integration/test_data_pipeline.py` and `tests/integration/test_training_pipeline.py`.
- **Reproducibility**: Set explicit seeds (`reproducibility.set_seed(42)`) in all stochastic functions, training routines, and data splits.
- **Running Tests**:
  ```bash
  pytest tests/ --cov=src/kidneycare --cov-report=term-missing
  ```

---

## 7. Submitting a Pull Request

1. Ensure your branch is rebased onto the latest `main`.
2. Run test and lint suites locally:
   ```bash
   make lint
   make test
   ```
3. Open a Pull Request referencing the corresponding issue (e.g., `Closes #12`).
4. Complete all items in the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
5. At least one maintainer and one clinical/technical reviewer must approve before merge.
