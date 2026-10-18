## Description
Provide a concise explanation of the changes made, the underlying problem addressed, and the clinical / engineering rationale.

## Type of Change
- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] ✨ New feature (non-breaking change adding functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 🏥 Clinical / Model adjustment (updates to biomarkers, thresholds, calibration, or baseline models)
- [ ] 📝 Documentation update (docs, README, tutorials, or docstrings)
- [ ] ⚙️ CI / CD / Build / Dependency update

## Clinical Safety & PHI Protection Verification
- [ ] **NO PHI / Patient Identifiers**: I confirm that no Protected Health Information, real patient data, or non-anonymized records are included in this PR or repository commits.
- [ ] **Physiological Bounds & Safety**: Any new feature thresholds or preprocessing rules comply with clinical nephrology references (e.g., KDIGO).
- [ ] **Reproducibility**: Random seeds and hardware determinism parameters are specified.

## Testing & Verification
- [ ] All unit tests pass locally (`pytest tests/unit/`)
- [ ] Integration pipelines pass (`pytest tests/integration/`)
- [ ] New unit tests have been added covering new functionality or bug fixes
- [ ] Code formatting and linting pass (`black --check`, `flake8`, `mypy`)

## Related Issues
Closes #
