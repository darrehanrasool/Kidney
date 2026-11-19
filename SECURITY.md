# Security & Clinical Data Protection Policy

## Healthcare Data & Privacy Compliance

Kidney-AI is designed for academic, clinical, and translational research. Maintaining strict data privacy and adherence to healthcare standards is paramount:

- **De-Identification**: All clinical data used with this platform must be thoroughly de-identified in strict adherence with HIPAA Privacy Rule Safe Harbor standards and GDPR Article 9 requirements.
- **No PHI in Version Control**: Under no circumstances should raw patient records, medical record numbers (MRNs), names, or unmasked healthcare identifiers be pushed to version control.
- **Data Segregation**: Training data, splits, and local caches must remain in `.gitignore`-protected directories (`data/raw/`, `data/processed/`, `data/splits/`).

## Reporting Vulnerabilities

If you discover any security vulnerability, dependency issue, or potential data leakage risk:

1. **Do not** disclose the vulnerability publicly in an open issue.
2. Email the maintainer directly at: `rehanrasool@iust.ac.in`.
3. Provide detailed steps to reproduce the issue and affected files/configurations.
4. We will acknowledge receipt within 48 hours and work with you to release a patch promptly.
