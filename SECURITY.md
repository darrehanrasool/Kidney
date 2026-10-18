# Security Policy

## Supported Versions

We release security patches and dependency updates for the following versions of **KidneyCare-AI**:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

---

## Reporting a Vulnerability

The KidneyCare-AI team takes security, software reliability, and data privacy seriously, especially given our focus on medical applications and healthcare analytics.

If you believe you have discovered a security vulnerability in KidneyCare-AI:

1. **Do NOT report security vulnerabilities via public GitHub issues.**
2. Send an email to the security response team at **[security@kidneycare-ai.org]**.
3. Include the following details in your report:
   - Description of the vulnerability and attack vector.
   - Proof of concept (PoC) or reproducible steps.
   - Affected files, versions, and dependencies.
   - Potential impact on clinical predictions, data integrity, or host environments.
4. You will receive an initial acknowledgment within 48 hours.
5. We will coordinate a responsible disclosure timeline with you once a patch is verified.

---

## Clinical Data & Patient Privacy (HIPAA / GDPR)

KidneyCare-AI operates under a strict **Zero-PHI (Protected Health Information) Policy**:

- **No Real Patient Data**: This repository does not host, store, or process identifiable patient data.
- **Data Scrubbing**: All datasets, demo notebooks, fixtures, and unit test mocks must be derived from de-identified benchmarks or purely synthetic distributions.
- **Accidental PHI Ingestion**: If identifiable clinical records are accidentally pushed or exposed in logs, contact `security@kidneycare-ai.org` immediately for permanent Git commit history purge and security revocation.

---

## Model Integrity & Adversarial Robustness

Because outputs may be evaluated in clinical decision-support contexts:
- **Out-of-Distribution (OOD) Protection**: Input validation layers check for extreme, physiologically impossible lab values (e.g., negative blood pressure or creatinine > 50 mg/dL) to prevent pipeline failures or silent corrupted outputs.
- **Model Checkpoint Signing**: Saved model weights under `checkpoints/` and `outputs/trained_models/` should have accompanying SHA-256 integrity hashes stored in `checkpoints/metadata/`.

---

## Supply Chain Security

- Maintainers actively monitor dependencies for known CVEs using automated Dependabot scans and GitHub secret scanners.
- Development builds should verify package checksums and use virtual environment isolation.
