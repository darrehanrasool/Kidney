# Scientific Literature & Research Benchmarking

This directory contains literature reviews, methodology extractions, baseline comparisons, and research gap taxonomies for Kidney-AI.

---

## 📚 Centralized Literature & Benchmarking Index

Full-text research manuscripts, benchmark studies, and dataset resources are cataloged in [`research_resources/`](../research_resources/README.md).

| # | Study Title | Authors | Journal / Year | Primary Contribution | Manuscript |
|---|-------------|---------|----------------|----------------------|------------|
| 1 | **Detection and diagnosis of CKD using deep learning-based HMANN** | Ma et al. | *FGCS* (2020) | Heterogeneous Modified ANN (HMANN), SVM + MLP-BP | [PDF](../research_resources/papers/Ma_et_al_2020_HMANN_CKD_Diagnosis.pdf) |
| 2 | **Detection and classification of kidney disease using CNNs** | Hannan & Pal | *J. Neurol Neurorehab Res* (2023) | Multi-layer 2D CNN with anisotropic diffusion preprocessing | [PDF](../research_resources/papers/Hannan_Pal_2023_CNN_Kidney_Disease.pdf) |
| 3 | **Diagnosing and Characterizing CKD with Machine Learning** | Figueroa et al. | *MDPI Electronics* (2024) | LightGBM, AutoML (`df-analyze`), clinical symptom importance | [PDF](../research_resources/papers/Figueroa_et_al_2024_Diagnosing_Characterizing_CKD_ML.pdf) |
| 4 | **Intelligent Diagnostic Prediction & Classification Models for CKD** | Poonia et al. | *MDPI Healthcare* (2022) | Chi-Square & RFE feature selection with Logistic Regression & ensembles | [PDF](../research_resources/papers/Poonia_et_al_2022_Intelligent_Diagnostic_Models_Kidney_Disease.pdf) |

---

## 📁 Directory Structure

```
papers/
├── README.md                 # Index of referenced nephrology and medical ML literature
├── baseline_comparisons/     # Benchmarking matrix against state-of-the-art models
├── literature_reviews/       # Structured surveys on CKD detection, AKI prediction & XAI
├── metadata/                 # Machine-readable publication metadata (JSON/YAML)
├── methodology_extraction/   # Mathematical formulations, eGFR equations, loss functions
├── pdf/                      # Local research paper PDFs and preprints
└── research_gaps/            # Unresolved challenges in clinical AI translation
```

For full literature summaries, abstracts, and dataset availability statements, see:
- 📖 [Research Papers Catalog](../research_resources/papers/README.md)
- 📊 [Dataset Sources & Access Links](../research_resources/dataset/README.md)
