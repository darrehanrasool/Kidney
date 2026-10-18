# KidneyCare-AI Research & Literature Repository

This directory contains the state-of-the-art research literature, academic papers (PDFs), extracted methodologies, mathematical formulations, baseline benchmarks, and open research gap analyses for chronic kidney disease detection.

## Directory Layout

- `pdf/`: Full-text research papers and clinical analysis atlases:
  - `Kidney.pdf`: Kidney AI Research Atlas: Comparative retrieval of methods, equations, datasets, and accuracies
  - `kidney-1.pdf` & `kidney-2.pdf`: Ma et al. (2020), *Future Generation Computer Systems*, "Detection and diagnosis of CKD using deep learning-based Heterogeneous Modified ANN" (FGCS 111:17-26)
  - `kidney-3.pdf`: Hannan & Pal (2023), "Detection and classification of kidney disease using convolution neural network"
  - `kidney-4.pdf`: Poonia et al. (2022), *Healthcare* (MDPI), "Intelligent Diagnostic Prediction and Classification Models for Detection of Kidney Disease"
- `metadata/`: Structured JSON catalog (`papers_metadata.json`) containing citation metadata, DOIs, architecture specifications, and reproducibility statuses.
- `literature_reviews/`: Systematic surveys comparing clinical tabular benchmarks, imaging modalities, and hybrid neural models (`ckd_machine_learning_survey.md`).
- `methodology_extraction/`: Mathematical equations, eGFR formulas, Chi-Square feature selection, RBF SVM, and loss functions (`mathematical_formulations.md`).
- `baseline_comparisons/`: Side-by-side benchmark comparison matrix evaluating reported metrics against locally reproduced models (`benchmark_comparison_matrix.md`).
- `research_gaps/`: Critical examination of data leakage, calibration neglect, and demographic bias in medical AI (`open_challenges.md`).
