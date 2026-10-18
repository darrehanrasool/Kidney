---
name: Feature Request
about: Suggest an idea, enhancement, or new capability for KidneyCare-AI
title: '[FEAT] '
labels: ['enhancement', 'discussion']
assignees: ''
---

## Problem Statement
A clear and concise description of what the problem is.
*Example: I am always struggling to evaluate model calibration across differing clinical eGFR tiers because...*

## Proposed Solution
A clear and concise description of what you want to happen. Include proposed APIs, architectures, or workflow modifications.

## Clinical, Algorithmic, or Scientific Justification
- **Clinical Relevance**: (e.g., alignment with KDIGO guidelines, CKD staging, AKI-to-CKD transition)
- **Algorithmic Baseline**: (e.g., SHAP tree explainer optimization, calibrated ensemble, multi-task learning)
- **Literature Reference** (if applicable): DOI, arXiv link, or paper citation

## Alternatives Considered
A clear and concise description of any alternative solutions or features you've considered.

## Impact on Existing Architecture
- [ ] Core configuration (`configs/`)
- [ ] Data ingestion & validation (`src/kidneycare/data/`)
- [ ] Models & training pipeline (`src/kidneycare/models/`, `training/`)
- [ ] Evaluation & clinical metrics (`src/kidneycare/evaluation/`)
- [ ] Explainability & visualization (`src/kidneycare/explainability/`, `visualization/`)
- [ ] Hardware acceleration & resource limits (`src/kidneycare/hardware/`)

## Additional Context
Add any other context, diagrams, or mockups about the feature request here.
